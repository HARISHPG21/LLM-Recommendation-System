import torch.nn as nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math

from torch import autocast
from peft import LoraConfig, get_peft_model
from utils import print_rank0


class LLMRec(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.args = args

        # 1. Load backbone with quantization and FlashAttention configs
        if 'bert' in args.backbone:
            from transformers import BertModel
            self.llm = BertModel.from_pretrained(args.root_path + args.backbone)
        elif 'opt' in args.backbone:
            from transformers import OPTModel
            self.llm = OPTModel.from_pretrained(args.root_path + args.backbone)
        elif 'flan' in args.backbone:
            from llm.modeling_t5 import T5EncoderModel
            self.llm = T5EncoderModel.from_pretrained(args.root_path + args.backbone)
        elif any(x in args.backbone.lower() for x in ['llama', 'gemma', 'mistral']):
            from transformers import AutoModel
            load_kwargs = {}
            if getattr(args, 'qlora', False):
                from transformers import BitsAndBytesConfig
                load_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            if getattr(args, 'flash_attn', False):
                load_kwargs['attn_implementation'] = "flash_attention_2"
            self.llm = AutoModel.from_pretrained(args.root_path + args.backbone, **load_kwargs)
        else:
            raise NotImplementedError(f"Backbone {args.backbone} not supported.")

        if args.lora:
            print_rank0("Initialize Lora From Scratch!", self.args.rank)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                use_dora=getattr(args, 'use_dora', False),
            )
            self.llm = get_peft_model(self.llm, config)
            self.trainable2float()

        self.item_embs = None

        # 2. Hybrid collaborative components
        if getattr(args, 'model_type', 'llm') == 'hybrid':
            self.id_embeddings = nn.Embedding(args.item_count, args.item_emb_dim, padding_idx=0)
            self.id_gru = nn.GRU(args.item_emb_dim, args.item_emb_dim, batch_first=True)
            
            # dynamically detect LLM hidden dimension
            llm_dim = getattr(self.llm.config, 'hidden_size', 768)
            self.llm_proj = nn.Linear(llm_dim, args.item_emb_dim)
            self.gate = nn.Linear(args.item_emb_dim * 2, 1)

    # 将可学习的参数都转换成float32，不然amp会出问题
    def trainable2float(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print_rank0(f"Trainable Parameter:{name}", self.args.rank)
                param.data = param.data.float()

    def get_embedding(self, input_ids, attention_mask):
        llm_output = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        if 'bert' == self.args.backbone[:4]:
            return llm_output[0][:, 0]
        elif 'opt' == self.args.backbone[:3] or any(x in self.args.backbone.lower() for x in ['llama', 'gemma', 'mistral']):
            return self.gather_indexes(llm_output.last_hidden_state, attention_mask.sum(dim=-1) - 1)
        elif 'flan' == self.args.backbone[:4]:
            return self.gather_indexes(llm_output.last_hidden_state, attention_mask.sum(dim=-1) - 1)
        else:
            raise NotImplementedError(f"Embedding extraction for {self.args.backbone} not supported.")

    def forward(self, inputs):
        device = next(self.parameters()).device
        seq_cls = self.get_embedding(input_ids=inputs['sequence_input_ids'], attention_mask=inputs['sequence_attention_mask'])
        item_cls = self.get_embedding(input_ids=inputs['item_input_ids'], attention_mask=inputs['item_attention_mask'])
        item_cls = item_cls.view(seq_cls.size()[0], self.args.train_nega_count + 1, item_cls.size()[-1])

        if getattr(self.args, 'model_type', 'llm') == 'hybrid':
            # Collaborative sequence representation
            seq_id_embs = self.id_embeddings(inputs['seq_iids'])
            gru_out, _ = self.id_gru(seq_id_embs)
            seq_lengths = (inputs['seq_iids'] != 0).sum(dim=-1) - 1
            seq_lengths = torch.clamp(seq_lengths, min=0)
            seq_id_rep = gru_out[torch.arange(gru_out.size(0)), seq_lengths]

            # Collaborative candidate item representations
            item_id_rep = self.id_embeddings(inputs['negative_items'])

            # Project LLM representations
            seq_llm_rep = self.llm_proj(seq_cls)
            item_llm_rep = self.llm_proj(item_cls)

            # Fuse user/sequence representation
            seq_gate = torch.sigmoid(self.gate(torch.cat([seq_id_rep, seq_llm_rep], dim=-1)))
            fused_seq = seq_gate * seq_id_rep + (1 - seq_gate) * seq_llm_rep

            # Fuse item representations
            flat_item_id = item_id_rep.view(-1, self.args.item_emb_dim)
            flat_item_llm = item_llm_rep.view(-1, self.args.item_emb_dim)
            item_gate = torch.sigmoid(self.gate(torch.cat([flat_item_id, flat_item_llm], dim=-1)))
            fused_item = item_gate * flat_item_id + (1 - item_gate) * flat_item_llm
            fused_item = fused_item.view(item_cls.size(0), item_cls.size(1), self.args.item_emb_dim)

            # Determine device type for autocast dynamically
            device_type = 'cuda' if next(self.parameters()).is_cuda else 'cpu'
            with autocast(device_type=device_type, enabled=False):
                fused_item = fused_item.float()
                fused_seq = fused_seq.float().unsqueeze(-1)
                scores = torch.bmm(fused_item, fused_seq).squeeze(-1)
                loss = F.cross_entropy(scores, inputs['target_position'])
            return [loss, loss]
        else:
            device_type = 'cuda' if next(self.parameters()).is_cuda else 'cpu'
            with autocast(device_type=device_type, enabled=False):
                item_cls = item_cls.float()
                seq_cls = seq_cls.float().unsqueeze(-1)
                scores = torch.bmm(item_cls, seq_cls).squeeze(-1)
                loss = F.cross_entropy(scores, inputs['target_position'])
            return [loss, loss]

    def valid_step(self, inputs):
        seq_cls = self.get_embedding(input_ids=inputs['sequence_input_ids'], attention_mask=inputs['sequence_attention_mask'])

        if getattr(self.args, 'model_type', 'llm') == 'hybrid':
            # Collaborative sequence representation
            seq_id_embs = self.id_embeddings(inputs['seq_iids'])
            gru_out, _ = self.id_gru(seq_id_embs)
            seq_lengths = (inputs['seq_iids'] != 0).sum(dim=-1) - 1
            seq_lengths = torch.clamp(seq_lengths, min=0)
            seq_id_rep = gru_out[torch.arange(gru_out.size(0)), seq_lengths]

            # Project LLM sequence representation
            seq_llm_rep = self.llm_proj(seq_cls)
            self.last_seq_id_rep = seq_id_rep.detach().cpu()
            self.last_seq_llm_rep = seq_llm_rep.detach().cpu()

            # Fuse
            seq_gate = torch.sigmoid(self.gate(torch.cat([seq_id_rep, seq_llm_rep], dim=-1)))
            self.last_gate_value = seq_gate.detach().cpu().tolist()
            seq_cls = seq_gate * seq_id_rep + (1 - seq_gate) * seq_llm_rep

        item_cls = self.item_embs[inputs['negative_items']].to(seq_cls.device)

        device_type = 'cuda' if next(self.parameters()).is_cuda else 'cpu'
        with autocast(device_type=device_type, enabled=False):
            item_cls = item_cls.float()
            seq_cls = seq_cls.float().unsqueeze(-1)
            scores = torch.bmm(item_cls, seq_cls).squeeze(-1) / math.sqrt(item_cls.size()[-1])
            loss = F.cross_entropy(scores, inputs['target_position'])

        return scores, inputs['target_position']

    @torch.no_grad()
    def generate_embs(self, item_tokens):
        # 3. Cache item embeddings across validation epochs
        if getattr(self.args, 'cache_item_embs', False) and hasattr(self, 'item_embs') and self.item_embs is not None:
            return

        if hasattr(self, 'item_embs') and self.item_embs is not None:
            del self.item_embs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print_rank0(f"GPU:{self.args.rank} Generating Embedding")
        item_ids = item_tokens['item_ids']
        item_attn = item_tokens['item_attn']
        device = next(self.parameters()).device

        item_embs = []
        batch_size = 128
        if self.args.rank == 0:
            iterator = tqdm(range(0, item_ids.size()[0], batch_size), desc='Generate embs')
        else:
            iterator = range(0, item_ids.size()[0], batch_size)
        for start_idx in iterator:
            batch_item_ids = item_ids[start_idx: start_idx + batch_size].to(device)
            batch_item_attn = item_attn[start_idx: start_idx + batch_size].to(device)
            batch_item_embs = self.get_embedding(input_ids=batch_item_ids, attention_mask=batch_item_attn)
            item_embs.append(batch_item_embs.detach())
        self.item_embs = torch.cat(item_embs, dim=0)
        assert self.item_embs.size()[0] == item_ids.size()[0]

        if getattr(self.args, 'model_type', 'llm') == 'hybrid':
            # Project LLM representations, fetch ID embeddings, and fuse
            proj_llm = self.llm_proj(self.item_embs)
            all_ids = torch.arange(proj_llm.size(0), device=device)
            id_embs = self.id_embeddings(all_ids)
            gate_val = torch.sigmoid(self.gate(torch.cat([id_embs, proj_llm], dim=-1)))
            fused_embs = gate_val * id_embs + (1 - gate_val) * proj_llm
            self.item_embs = fused_embs

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)
