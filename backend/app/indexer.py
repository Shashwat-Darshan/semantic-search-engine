# backend/app/indexer.py
from elasticsearch import Elasticsearch, helpers
from .config import ELASTIC_URL, INDEX_NAME, EMBED_DIM
from .config import 
from .config import 

from .config import ELASTIC_URL, INDEX_NAME
from .config import EMBED_DIM

from elasticsearch import Elasticsearch, helpers
from .config import ELASTIC_URL, INDEX_NAME, EMBED_DIM

es = Elasticsearch(ELASTIC_URL)

def create_index():
    settings = {
        "mappings": {
            "properties": {
                "title":     {"type": "text"},
                "content":   {"type": "text"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBED_DIM,
                    "index": True,
                    "similarity": "cosine"
                },
                "tags":      {"type": "keyword"},
                "timestamp": {"type": "date"}
            }
        }
    }
    es.indices.create(index=INDEX_NAME, ignore=400, body=settings)

def index_documents(docs):
    """
    docs: iterable of dicts with keys id, title, content, embedding, tags, timestamp
    """
    actions = []
    for d in docs:
        action = {
            "_index": INDEX_NAME,
            "_id":    d["id"],
            "_source": {
                "title":     d["title"],
                "content":   d["content"],
                "embedding": d["embedding"].tolist(),
                "tags":      d.get("tags", []),
                "timestamp": d.get("timestamp")
            }
        }
        actions.append(action)
    helpers.bulk(es, actions)
