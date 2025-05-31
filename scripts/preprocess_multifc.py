import pandas as pd
import json

def preprocess_multifc():
    # File paths
    input_file = "data/raw/multifc/claims.csv"
    csv_output = "data/processed/multifc/claims_processed.csv"
    json_output = "data/processed/multifc/claims_processed.json"
    
    print("📖 Loading MultiFC dataset...")
    df = pd.read_csv(input_file)
    
    # Drop rows with missing 'claim' or 'label' (essential fields)
    df = df.dropna(subset=['claim', 'label'])
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Combine claim and reason into a single text column (handle missing reasons)
    df['reason'] = df['reason'].fillna('')
    df['text'] = df['claim'] + " " + df['reason']
    
    # Map labels to binary (example mapping, adjust if your task needs)
    label_mapping = {
        'Supported': 1,
        'Refuted': 0,
        'Not Enough Info': 0
    }
    df['label'] = df['label'].map(label_mapping).fillna(0).astype(int)
    
    # Keep only relevant columns for further processing
    df_processed = df[['claimID', 'text', 'label', 'categories', 'speaker', 'publish date']]
    
    # Save to CSV
    print(f"💾 Saving cleaned CSV to {csv_output}")
    df_processed.to_csv(csv_output, index=False)
    
    # Save to JSON
    print(f"💾 Saving cleaned JSON to {json_output}")
    records = df_processed.to_dict(orient='records')
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2)
    
    # Print sample for verification
    print("\nSample of processed MultiFC data:")
    print(df_processed.head(3))
    print(f"\nTotal rows processed: {len(df_processed)}")

if __name__ == "__main__":
    preprocess_multifc()
