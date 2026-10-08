"""In-memory hybrid index: dense cosine similarity + BM25 keyword scoring, with on-disk persistence.

A NumPy matrix is enough for tens of thousands of chunks (a matrix-vector product
over 20k x 384 floats takes ~1 ms). Swapping in FAISS or a vector DB later only
changes this file.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunking import Chunk


# keep dots, dashes and underscores so "config.pbtxt" and "gpu-memory-utilization" stay one token
_TOKEN = re.compile(r"[a-z0-9][a-z0-9_.\-]*[a-z0-9]|[a-z0-9]")


def tokenize(text: str) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    # also index the parts of compound tokens: "gpu-memory-utilization" -> gpu, memory, utilization
    return toks + [p for t in toks if re.search(r"[_.\-]", t) for p in re.split(r"[_.\-]+", t) if p]


class BM25:
    """Classic Okapi BM25 over the chunk texts. Built in memory from the chunks (fast, no model)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = np.zeros(self.n, dtype=np.float32)
        for i, d in enumerate(docs):
            tf = Counter(tokenize(d))
            self.lengths[i] = sum(tf.values())
            for term, c in tf.items():
                self.postings[term].append((i, c))
        self.avgdl = float(self.lengths.mean()) if self.n else 0.0

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            idx = np.fromiter((i for i, _ in plist), dtype=np.int64, count=len(plist))
            tf = np.fromiter((c for _, c in plist), dtype=np.float32, count=len(plist))
            norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[idx] / self.avgdl)
            out[idx] += idf * tf * (self.k1 + 1) / norm
        return out


def _ranks(scores: np.ndarray, depth: int) -> list[int]:
    depth = min(depth, len(scores))
    top = np.argpartition(-scores, depth - 1)[:depth]
    return list(top[np.argsort(-scores[top])])


class VectorIndex:
    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.matrix = np.zeros((0, 1), dtype=np.float32)
        self.bm25: BM25 | None = None

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.matrix = self.embedder.embed([c.text for c in chunks])
        self.bm25 = BM25([c.text for c in chunks])

    def search(self, query: str, k: int = 4, mode: str = "hybrid") -> list[tuple[Chunk, float]]:
        """mode = dense | bm25 | hybrid. Hybrid merges the dense and BM25 rankings with
        Reciprocal Rank Fusion (score = sum of 1/(60 + rank)), which needs no score calibration.
        The returned score is always the dense cosine similarity, so it stays interpretable."""
        if not self.chunks:
            return []
        k = min(k, len(self.chunks))
        dense = self.matrix @ self.embedder.embed([query])[0]
        if mode == "dense" or self.bm25 is None:
            order = _ranks(dense, k)
        elif mode == "bm25":
            order = _ranks(self.bm25.scores(query), k)
        else:
            fused: dict[int, float] = defaultdict(float)
            for ranking in (_ranks(dense, 50), _ranks(self.bm25.scores(query), 50)):
                for r, i in enumerate(ranking):
                    fused[int(i)] += 1.0 / (60 + r)
            order = sorted(fused, key=fused.get, reverse=True)[:k]
        return [(self.chunks[i], float(dense[i])) for i in order]

    # persistence: lets pods restart without re-embedding the whole corpus
    def save(self, directory: str | Path, fingerprint: str = "") -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "matrix.npy", self.matrix)
        (d / "chunks.json").write_text(
            json.dumps(
                {"embedder": self.embedder.name, "fingerprint": fingerprint, "chunks": [asdict(c) for c in self.chunks]}
            )
        )

    def load(self, directory: str | Path, fingerprint: str | None = None) -> bool:
        """Load a saved index. Returns False (caller rebuilds) if it is missing, was built with a
        different embedding model, or - when a fingerprint is given - from different docs."""
        d = Path(directory)
        if not (d / "matrix.npy").exists() or not (d / "chunks.json").exists():
            return False
        meta = json.loads((d / "chunks.json").read_text())
        if meta.get("embedder") != self.embedder.name:
            return False  # index was built with a different model; rebuild
        if fingerprint is not None and meta.get("fingerprint") != fingerprint:
            return False  # docs changed since the index was built
        self.matrix = np.load(d / "matrix.npy")
        self.chunks = [Chunk(**c) for c in meta["chunks"]]
        self.bm25 = BM25([c.text for c in self.chunks])
        return True
