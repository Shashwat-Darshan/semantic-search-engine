# backend/app/utils.py
import json
import re

def load_raw_data(path):
    """Yield dicts from a JSONL file."""
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)

def clean_text(text):
    """Basic text cleaning."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, max_words=200):
    """Split long text into ~max_words chunks."""
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i : i + max_words])
