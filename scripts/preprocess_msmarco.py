import os
import json
import pandas as pd
import ast

RAW_DIR = "data/raw/msMarco"
PROCESSED_DIR = "data/processed/msmarco"

def preprocess_train_subset(path):
    print(f"Processing train_subset from {path} ...")
    df = pd.read_json(path, lines=True)
    print(f"✅ Processed train_subset → {len(df)} rows")
    return df

def preprocess_usefulness(path):
    print(f"Processing Usefulness.tsv from {path} ...")
    df = pd.read_csv(path, sep='\t', engine='python')
    df['labelhistogram'] = df['labelhistogram'].apply(ast.literal_eval)
    df['Useful_count'] = df['labelhistogram'].apply(lambda d: d.get('Useful', 0))
    print(f"✅ Processed Usefulness.tsv → {len(df)} rows")
    return df

def preprocess_sessions_variable_columns(path):
    print(f"Processing sessions file with variable columns from {path} ...")
    df = pd.read_csv(path, sep='\t', header=None, engine='python', on_bad_lines='skip')
    df['query_response'] = df.iloc[:, 1:].astype(str).agg(' '.join, axis=1).str.strip()
    df.rename(columns={0: 'session_id'}, inplace=True)
    df_clean = df[['session_id', 'query_response']]
    df_clean = df_clean[df_clean['query_response'] != '']
    df_clean = df_clean.reset_index(drop=True)
    print(f"✅ Processed sessions → {len(df_clean)} rows")
    return df_clean

def main():
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    train_path = os.path.join(RAW_DIR, "train_subset.jsonl")
    df_train = preprocess_train_subset(train_path)
    df_train.to_csv(os.path.join(PROCESSED_DIR, "train_subset.csv"), index=False)

    usefulness_path = os.path.join(RAW_DIR, "Usefulness.tsv")
    df_useful = preprocess_usefulness(usefulness_path)
    df_useful.to_csv(os.path.join(PROCESSED_DIR, "Usefulness_processed.csv"), index=False)

    sessions_path = os.path.join(RAW_DIR, "artificialSessionsBERT500k.tsv")
    df_sessions = preprocess_sessions_variable_columns(sessions_path)
    df_sessions.to_csv(os.path.join(PROCESSED_DIR, "sessions_cleaned.csv"), index=False)

    sessions_json_path = os.path.join(PROCESSED_DIR, "sessions_cleaned.json")
    with open(sessions_json_path, 'w', encoding='utf-8') as f:
        json.dump(df_sessions.to_dict(orient='records'), f, indent=2)

    print("✅ All files processed and saved successfully.")

if __name__ == "__main__":
    main()
