# backend/app/api.py
from fastapi import FastAPI, HTTPException, Query
from elasticsearch import Elasticsearch
import numpy as np

from .config import ELASTIC_URL, INDEX_NAME
from .embedder import embed_text
from .indexer import create_index
from .cache import get_cached, set_cached

app = FastAPI()
es = Elasticsearch(ELASTIC_URL)

@app.on_event("startup")
async def startup_event():
    create_index()

@app.get("/search")
async def search(q: str = Query(..., min_length=1), k: int = 5):
    # Check cache
    cached = await get_cached(q)
    if cached:
        return {"results": cached}

    # Embed query
    q_vec = embed_text([q])[0].tolist()

    # Elasticsearch k-NN query
    body = {
        "size": k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": q_vec}
                }
            }
        }
    }
    res = es.search(index=INDEX_NAME, body=body)
    hits = res["hits"]["hits"]
    results = [
        {
            "id":    h["_id"],
            "score": h["_score"],
            "title": h["_source"]["title"],
            "snippet": h["_source"]["content"][:200]
        }
        for h in hits
    ]

    # Cache and return
    await set_cached(q, results)
    return {"results": results}
