"""Retriever microservice: owns the document index and answers similarity searches."""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .chunking import corpus_fingerprint, load_corpus
from .embeddings import make_embedder
from .index import VectorIndex

DOCS_DIR = os.getenv("DOCS_DIR", "data/docs")
INDEX_DIR = os.getenv("INDEX_DIR", "data/index")

state: dict = {}


def build_index(force: bool = False) -> dict:
    index: VectorIndex = state["index"]
    t0 = time.perf_counter()
    fp = corpus_fingerprint(DOCS_DIR)
    if not force and index.load(INDEX_DIR, fingerprint=fp):
        source = "disk"
    else:
        index.build(load_corpus(DOCS_DIR))
        index.save(INDEX_DIR, fingerprint=fp)
        source = "built"
    stats = {
        "chunks": len(index.chunks),
        "files": len({c.source for c in index.chunks}),
        "embedder": index.embedder.name,
        "source": source,
        "seconds": round(time.perf_counter() - t0, 2),
    }
    state["stats"] = stats
    print(f"[retriever] index ready: {stats}")
    return stats


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["index"] = VectorIndex(make_embedder())
    build_index()
    yield


app = FastAPI(title="retriever", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=20)
    mode: Literal["hybrid", "dense", "bm25"] = Field(default_factory=lambda: os.getenv("RETRIEVAL_MODE", "hybrid"))


class Hit(BaseModel):
    id: str
    source: str
    section: str
    text: str
    score: float


class SearchResponse(BaseModel):
    hits: list[Hit]
    took_ms: float


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    t0 = time.perf_counter()
    hits = [
        Hit(id=c.id, source=c.source, section=c.section, text=c.text, score=round(s, 4))
        for c, s in state["index"].search(req.query, req.k, req.mode)
    ]
    return SearchResponse(hits=hits, took_ms=round((time.perf_counter() - t0) * 1000, 2))


@app.post("/reindex")
def reindex() -> dict:
    return build_index(force=True)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": "index" in state, **state.get("stats", {})}
