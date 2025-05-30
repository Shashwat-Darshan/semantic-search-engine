from datasets import load_dataset
import json

ds = load_dataset("squad", split="train")
with open("data/raw/dataset.jsonl", "w") as f:
    for i, entry in enumerate(ds):
        doc = {
            "id":      f"squad_{i:06}",
            "title":   entry["title"],
            "content": entry["context"],
            "tags":    ["squad"],
            "timestamp": None
        }
        f.write(json.dumps(doc) + "\n")