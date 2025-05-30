# backend/app/embedder.py
from sentence_transformers import SentenceTransformer
from .config import EMBED_MODEL

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def embed_text(texts):
    """
    texts: list of str
    returns: list of vectors
    """
    model = get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
