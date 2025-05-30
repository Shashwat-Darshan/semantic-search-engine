# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from backend.app.api import app

client = TestClient(app)

def test_search_empty():
    resp = client.get("/search", params={"q": "", "k":5})
    assert resp.status_code == 422  # Validation error

def test_search_valid(monkeypatch):
    # Mock embed and ES search
    monkeypatch.setattr("backend.app.api.get_cached", lambda q: None)
    def fake_search(index, body): 
        return {"hits": {"hits": [
            {"_id": "doc_1", "_score": 1.23, "_source": {"title":"T","content":"C"}}
        ]}}
    monkeypatch.setattr("backend.app.api.es.search", fake_search)
    resp = client.get("/search", params={"q": "test", "k":1})
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"][0]["id"] == "doc_1"
