"""Embedding backends.

- FastEmbedEmbedder: real semantic embeddings (BAAI/bge-small-en-v1.5, ONNX, runs fine on CPU).
- HashEmbedder: dependency-free bag-of-words hashing. Used in tests and as a fallback
  so the system still runs if the model can't be downloaded.
"""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np

TOKEN = re.compile(r"[a-z0-9_]+")


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class HashEmbedder:
    name = "hash"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        tokens = TOKEN.findall(text.lower())
        for tok in tokens + [a + "_" + b for a, b in zip(tokens, tokens[1:])]:
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
            v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        return v

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._vec(t) for t in texts])) if texts else np.zeros((0, self.dim), np.float32)


class FastEmbedEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding  # imported lazily so tests don't need it

        self.name = model
        self._model = TextEmbedding(model_name=model)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), np.float32)
        return _normalize(np.array(list(self._model.embed(texts, batch_size=int(os.getenv('EMBED_BATCH', '16')))), dtype=np.float32))


def make_embedder():
    kind = os.getenv("EMBEDDER", "fastembed")
    if kind == "hash":
        return HashEmbedder()
    try:
        return FastEmbedEmbedder(os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    except Exception as exc:  # model download blocked, package missing, etc.
        print(f"[retriever] fastembed unavailable ({exc!r}); falling back to hash embeddings")
        return HashEmbedder()
