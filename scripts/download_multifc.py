import os
import pandas as pd
from datasets import load_dataset

SAVE_DIR = "data/raw/multifc"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_multifc():
    print("[+] Downloading MultiFC dataset from Hugging Face...")
    try:
        # Load dataset from Hugging Face
        dataset = load_dataset("pszemraj/multi_fc")
        
        # Convert to pandas DataFrame and save
        df = pd.DataFrame(dataset['train'])
        output_path = os.path.join(SAVE_DIR, "claims.csv")
        df.to_csv(output_path, index=False)
        
        print(f"[+] Saved claims.csv to {SAVE_DIR}")
        return True
    except Exception as e:
        print(f"[!] Error during download: {e}")
        return False

def preview_sample():
    path = os.path.join(SAVE_DIR, "claims.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        print("\nDataset Information:")
        print(f"Number of columns: {len(df.columns)}")
        print("\nColumn names:")
        for col in df.columns:
            print(f"- {col}")
        print("\nPreview of the dataset:")
        print(df.head())
    else:
        print("[!] File not found.")

if __name__ == "__main__":
    # First install required package:
    # pip install datasets
    if download_multifc():
        print("[+] Download successful.")
    else:
        print("[!] Download failed.")
    preview_sample()
