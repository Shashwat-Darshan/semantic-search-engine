# backend/app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

ELASTIC_URL = os.getenv("ELASTIC_URL", "http://elasticsearch:9200")
INDEX_NAME  = os.getenv("INDEX_NAME", "documents")
REDIS_URL   = os.getenv("REDIS_URL", "redis://redis:6379/0")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM   = int(os.getenv("EMBED_DIM", 384))
