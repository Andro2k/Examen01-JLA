"""FastAPI endpoints for the SciFact retriever."""

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from examen01_jla.data import load_scifact
from examen01_jla.retrieval import SearchIndex

app = FastAPI(title="SciFact Information Retrieval", version="1.0.0")


@lru_cache(maxsize=1)
def get_index() -> SearchIndex:
    documents, _, _ = load_scifact("test")
    return SearchIndex(documents)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    model: str = "bm25"
    top_k: int = Field(default=10, ge=1, le=100)


@app.get("/health")
def health():
    return {"status": "ok", "corpus": "SciFact"}


@app.post("/search")
def search(request: SearchRequest):
    model = request.model.lower()
    if model not in {"bm25", "tfidf"}:
        raise HTTPException(status_code=400, detail="model debe ser 'bm25' o 'tfidf'")
    try:
        return {"model": model, "results": get_index().search(request.query, model, request.top_k)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
