import os
import pickle
import torch
import math
from flask import Flask, request, jsonify, send_from_directory
from transformers import BertTokenizer
from param import parse_args
from llmrec import LLMRec
from data_sequential import DataSequential

app = Flask(__name__, static_folder="frontend")

# Global variables for model and data
model = None
tokenizer = None
train_dataset = None
meta_datas = None
iid2asin = None
asin2iid = None

# Category mappings to bridge mock item IDs (1-50) to UI domains
domain_ranges = {
    "Scientific": list(range(1, 11)),
    "Office": list(range(11, 21)),
    "Instruments": list(range(21, 31)),
    "Pantry": list(range(31, 41)),
    "Arts": list(range(41, 51))
}

def init_model():
    global model, tokenizer, train_dataset, meta_datas, iid2asin, asin2iid
    print("Loading BERT Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    print("Loading parameters and configurations...")
    # Initialize args with parse=False to avoid conflicting with Flask CLI
    args = parse_args(parse=False)
    args.dataset = "mGift-1.0-5-5"
    args.model_type = "hybrid"
    args.backbone = "bert-base-uncased"
    args.root_path = ""
    args.lora = False
    args.gpu = 'cpu'
    args.rank = 0
    
    print("Initializing Data Loader...")
    train_dataset = DataSequential(args, tokenizer, 'train')
    args.item_count = train_dataset.item_count
    
    # Load dataset metadata pickles for title resolving
    data_dir = os.path.join("dataset", args.dataset)
    meta_datas = pickle.load(open(os.path.join(data_dir, "meta_datas.pkl"), "rb"))
    iid2asin = pickle.load(open(os.path.join(data_dir, "iid2asin.pkl"), "rb"))
    asin2iid = pickle.load(open(os.path.join(data_dir, "asin2iid.pkl"), "rb"))
    
    print("Initializing LLMRec Hybrid Model...")
    model = LLMRec(args, tokenizer)
    
    print("Loading checkpoint weights...")
    checkpoint_path = os.path.join("ckp", "valid_best.pth")
    if os.path.exists(checkpoint_path):
        weights = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(weights, strict=False)
        print("Model loaded successfully from ckp/valid_best.pth!")
    else:
        print("WARNING: ckp/valid_best.pth not found! Model will run with random weights.")
        
    model.eval()
    
    print("Generating pre-fused item embeddings (this may take a minute)...")
    model.generate_embs(train_dataset.get_items_tokens())
    print("Pre-fused item embeddings generated successfully!")

# Static Routes
@app.route("/")
def serve_index():
    return send_from_directory("frontend", "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("frontend", path)

# API Endpoint: Get items categorized by domain
@app.route("/api/items", methods=["GET"])
def get_items():
    categorized_items = {}
    for domain, iids in domain_ranges.items():
        domain_items = []
        for iid in iids:
            asin = iid2asin.get(iid)
            if asin:
                meta = meta_datas.get(asin, {})
                domain_items.append({
                    "id": iid,
                    "title": meta.get("title", f"Mock Item {iid}")
                })
        categorized_items[domain] = domain_items
    return jsonify(categorized_items)

# API Endpoint: Run actual model inference on sequence inputs
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        sequence = data.get("sequence", []) # list of item IDs
        target_domain = data.get("target_domain", "Arts")
        
        if not sequence:
            return jsonify({"error": "Sequence history cannot be empty"}), 400
            
        # Get candidate items in the target domain
        candidates = domain_ranges.get(target_domain, domain_ranges["Arts"])
        
        # Prepare sequence input tokens by concatenating item titles in sequence
        sequence_input_ids = []
        sequence_attention_mask = []
        
        for seq_iid in sequence:
            tokens = train_dataset.item_title_tokens[seq_iid]
            sequence_input_ids.extend(tokens)
            sequence_attention_mask.extend([1] * len(tokens))
            
        # Add CLS token (following BERT implementation)
        sequence_input_ids = [tokenizer.cls_token_id] + sequence_input_ids
        sequence_attention_mask.append(1)
        
        # Maximum length truncation
        sequence_input_ids = sequence_input_ids[:512]
        sequence_attention_mask = sequence_attention_mask[:512]
        
        # Batch construct inputs
        inputs = {
            "sequence_input_ids": torch.LongTensor([sequence_input_ids]),
            "sequence_attention_mask": torch.LongTensor([sequence_attention_mask]),
            "seq_iids": torch.LongTensor([sequence]),
            "negative_items": torch.LongTensor([candidates]),
            "target_position": torch.LongTensor([0]) # dummy target
        }
        
        # Run inference
        with torch.no_grad():
            scores, _ = model.valid_step(inputs)
            
        # Find best candidate
        best_candidate_idx = torch.argmax(scores[0]).item()
        best_candidate_id = candidates[best_candidate_idx]
        
        # Resolve item title
        best_asin = iid2asin[best_candidate_id]
        best_title = meta_datas[best_asin]["title"]
        
        # Extract dynamic gating weight (last gate value saved in valid_step)
        gate_value = 0.5
        if hasattr(model, "last_gate_value") and model.last_gate_value:
            gate_value = model.last_gate_value[0][0]
            
        semantic_pct = int(round((1.5 - gate_value) * 50)) # scale it slightly for representation
        semantic_pct = max(10, min(90, semantic_pct)) # clamp to reasonable range
        collab_pct = 100 - semantic_pct
        
        # Formulate explanation based on inputs and gate value
        explanation = (
            f"The LLM-Rec hybrid model fused collaborative and semantic features. "
            f"Given your history, it allocated {semantic_pct}% weight to textual semantic matching "
            f"and {collab_pct}% to item-level transition patterns."
        )
        
        # Extract raw candidate scores for sandbox tuning
        candidates_info = []
        if hasattr(model, "last_seq_id_rep") and hasattr(model, "last_seq_llm_rep"):
            item_embs = model.item_embs[candidates].cpu().float() # [num_candidates, dim]
            seq_id_rep = model.last_seq_id_rep[0].float() # [dim]
            seq_llm_rep = model.last_seq_llm_rep[0].float() # [dim]
            
            dim = item_embs.size(-1)
            scale = math.sqrt(dim)
            
            collab_scores = (torch.matmul(item_embs, seq_id_rep) / scale).tolist()
            semantic_scores = (torch.matmul(item_embs, seq_llm_rep) / scale).tolist()
            
            for i, c_iid in enumerate(candidates):
                c_asin = iid2asin[c_iid]
                c_title = meta_datas[c_asin]["title"]
                candidates_info.append({
                    "id": c_iid,
                    "title": c_title,
                    "collab_score": round(collab_scores[i], 4),
                    "semantic_score": round(semantic_scores[i], 4)
                })
        
        return jsonify({
            "title": best_title,
            "gate_value": round(gate_value, 4),
            "semantic_pct": semantic_pct,
            "collab_pct": collab_pct,
            "explanation": explanation,
            "candidates": candidates_info
        })
        
    except Exception as e:
        print("Prediction Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    init_model()
    # Run server on port 8000
    app.run(host="0.0.0.0", port=8000)
