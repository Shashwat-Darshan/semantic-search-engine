import os
import requests
import json
from time import sleep

SAVE_PATH = "data/raw/openalex/"
os.makedirs(SAVE_PATH, exist_ok=True)

SEARCH_TERMS = [
    "artificial intelligence", "machine learning", "deep learning",
    "neural networks", "transformer model", "language models"
]

def fetch_openalex(term, per_page=10):
    url = f"https://api.openalex.org/works?filter=title.search:{term}&per-page={per_page}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print(f"[+] Fetched for: {term}")
            return data['results']
        else:
            print(f"[!] Error for {term}: {response.status_code}")
            return []
    except Exception as e:
        print(f"[!] Failed request: {e}")
        return []

def main():
    for term in SEARCH_TERMS:
        results = fetch_openalex(term)
        filename = term.replace(" ", "_") + ".json"
        path = os.path.join(SAVE_PATH, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        sleep(1)  # polite pause between API calls

if __name__ == "__main__":
    main()
