import json
from pathlib import Path
import pandas as pd
import requests
from typing import List, Dict
import time

# Step 1: Download function (if applicable)
def download_openalex_data(save_dir: Path):
    """Download sample data from OpenAlex API"""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # OpenAlex API endpoint for recent works
    url = "https://api.openalex.org/works"
    params = {
        "filter": "has_abstract:true",
        "per_page": 100,
        "sample": 100
    }
    
    try:
        print("[+] Downloading OpenAlex sample dataset...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Save raw response
        output_file = save_dir / "sample_works.json"
        with open(output_file, "w", encoding="utf-8") as f:
            for work in response.json()["results"]:
                # Save each work as a separate JSON line
                json_line = json.dumps({
                    "title": work.get("title", ""),
                    "abstract": work.get("abstract", "")
                })
                f.write(json_line + "\n")
        print(f"[+] Saved raw data to {output_file}")
        
    except Exception as e:
        print(f"[!] Error downloading data: {e}")

# Step 2 + 3: Load and preprocess combined
def load_and_preprocess_openalex(raw_dir: Path) -> List[Dict]:
    records = []
    try:
        for file in raw_dir.glob("*.json"):
            print(f"[+] Processing {file.name}")
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            paper = json.loads(line)
                            title = paper.get("title", "").strip()
                            abstract = paper.get("abstract", "").strip()
                            if title and abstract:
                                records.append({"title": title, "abstract": abstract})
                        except json.JSONDecodeError as e:
                            print(f"[!] Error parsing JSON line: {e}")
                            continue
    except Exception as e:
        print(f"[!] Error processing files: {e}")
    
    return records

# Step 4: Save processed data
def save_processed_data(records: List[Dict], processed_dir: Path):
    """Save processed records to CSV and JSONL formats"""
    processed_dir.mkdir(parents=True, exist_ok=True)
    csv_path = processed_dir / "openalex_processed.csv"
    jsonl_path = processed_dir / "openalex_processed.jsonl"

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Saved processed OpenAlex data to:\n  {csv_path}\n  {jsonl_path}")

def main():
    raw_data_dir = Path("data/raw/openalex")
    processed_data_dir = Path("data/processed/openalex")

    # Download sample data
    download_openalex_data(raw_data_dir)
    
    # Wait briefly to ensure file is written
    time.sleep(1)
    
    # Process the downloaded data
    records = load_and_preprocess_openalex(raw_data_dir)
    if records:
        save_processed_data(records, processed_data_dir)
        print(f"✅ OpenAlex dataset loaded and preprocessed: {len(records)} records")
    else:
        print("[!] No records were processed")

if __name__ == "__main__":
    main()
