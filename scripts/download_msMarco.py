import json
import pandas as pd
import os
from pathlib import Path
import ast


def load_train_subset_jsonl(path):
    try:
        df = pd.read_json(path, lines=True)
        print(f"✅ Loaded {len(df)} records from train_subset.jsonl")
        print("🧠 Columns found:", df.columns.tolist())
        return df
    except Exception as e:
        print("❌ Failed to load train_subset.jsonl:", e)
        return pd.DataFrame()

def load_usefulness_tsv(path):
    try:
        df = pd.read_csv(path, sep='\t', names=["query", "suggestion", "modelabel", "labelhistogram"], header=0)
        
        # Convert labelhistogram from string to dictionary safely
        df["labelhistogram"] = df["labelhistogram"].apply(ast.literal_eval)
        
        print(f"✅ Loaded {len(df)} records from Usefulness.tsv")
        print("🧠 Columns:", df.columns.tolist())
        print("🔍 Sample row:\n", df.iloc[0])
        return df
    except Exception as e:
        print("❌ Failed to load Usefulness.tsv:", e)
        return pd.DataFrame()


def load_artificial_sessions_tsv(path):
    try:
        df = pd.read_csv(path, sep='\t', names=["session_id", "query", "response"], on_bad_lines='skip', engine='python')
        print(f"✅ Loaded {len(df)} artificial sessions from TSV")
        print("🧠 Sample rows:\n", df.head(3))
        return df
    except Exception as e:
        print("❌ Failed to load artificial sessions:", e)
        return pd.DataFrame()



def main():
    path_base = "../data/raw/msMarco"

    df_train = load_train_subset_jsonl(f"{path_base}/train_subset.jsonl")
    df_usefulness = load_usefulness_tsv(f"{path_base}/Usefulness.tsv")
    df_sessions = load_artificial_sessions_tsv(f"{path_base}/artificialSessionsBERT500k.tsv")



if __name__ == "__main__":
    main()
