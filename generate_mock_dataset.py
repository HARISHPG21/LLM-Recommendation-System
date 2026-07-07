import os
import pickle
import torch
import random

def generate_mock():
    dataset_name = "mGift-1.0-5-5"
    data_dir = os.path.join("dataset", dataset_name)
    os.makedirs(data_dir, exist_ok=True)

    num_items = 1100
    num_users = 10
    nega_count = 1000

    # 1. asin2iid and iid2asin
    asin2iid = {f"asin_{i}": i for i in range(1, num_items + 1)}
    iid2asin = {i: f"asin_{i}" for i in range(1, num_items + 1)}

    # 2. meta_datas
    meta_datas = {
        f"asin_{i}": {
            "asin": f"asin_{i}",
            "title": f"Mock Product Title {i}",
            "dataset": "Gift_Cards"
        } for i in range(1, num_items + 1)
    }

    # 3. review_datas
    # Each user has 6 reviews so they have enough interactions for train, valid, test splits.
    review_datas = {}
    for u in range(1, num_users + 1):
        user_id = f"user_{u}"
        user_reviews = []
        # Let's give each user 6 interactions
        for i in range(6):
            item_id = u * 5 + i + 1  # different items for different users
            user_reviews.append([item_id, "5.0", "Gift_Cards", str(100 + i)])
        review_datas[user_id] = user_reviews

    # 4. single_domain_iid
    single_domain_iid = {"Gift_Cards": list(range(1, num_items + 1))}

    # 5. uid2rid and rid2uid
    uid2rid = {u: f"user_{u}" for u in range(1, num_users + 1)}
    rid2uid = {f"user_{u}": u for u in range(1, num_users + 1)}

    # 6. negatives
    random.seed(42)
    valid_negatives = []
    test_negatives = []
    for u in range(1, num_users + 1):
        # sample negatives from all items
        valid_negatives.append(random.sample(range(1, num_items + 1), nega_count))
        test_negatives.append(random.sample(range(1, num_items + 1), nega_count))

    # Save all files
    with open(os.path.join(data_dir, "asin2iid.pkl"), "wb") as f:
        pickle.dump(asin2iid, f)
    with open(os.path.join(data_dir, "iid2asin.pkl"), "wb") as f:
        pickle.dump(iid2asin, f)
    with open(os.path.join(data_dir, "meta_datas.pkl"), "wb") as f:
        pickle.dump(meta_datas, f)
    with open(os.path.join(data_dir, "review_datas.pkl"), "wb") as f:
        pickle.dump(review_datas, f)
    with open(os.path.join(data_dir, "single_domain_iid.pkl"), "wb") as f:
        pickle.dump(single_domain_iid, f)
    with open(os.path.join(data_dir, "uid2rid.pkl"), "wb") as f:
        pickle.dump(uid2rid, f)
    with open(os.path.join(data_dir, "rid2uid.pkl"), "wb") as f:
        pickle.dump(rid2uid, f)
    with open(os.path.join(data_dir, "negatives_valid-1000.pkl"), "wb") as f:
        pickle.dump(torch.LongTensor(valid_negatives), f)
    with open(os.path.join(data_dir, "negatives_test-1000.pkl"), "wb") as f:
        pickle.dump(torch.LongTensor(test_negatives), f)

    print(f"Mock dataset generated successfully in {data_dir}!")

if __name__ == "__main__":
    generate_mock()
