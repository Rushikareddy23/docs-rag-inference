"""Build the vector index ahead of time:  python -m retriever.build_index
The Dockerfile runs this so pods start in seconds instead of re-embedding every boot.
Skips the work if INDEX_DIR already holds an index of exactly these docs (pass --force to rebuild)."""
from __future__ import annotations

import os
import time

import sys

from .chunking import corpus_fingerprint, load_corpus
from .embeddings import make_embedder
from .index import VectorIndex

if __name__ == "__main__":
    t0 = time.perf_counter()
    docs, out = os.getenv("DOCS_DIR", "data/docs"), os.getenv("INDEX_DIR", "data/index")
    idx = VectorIndex(make_embedder())
    fp = corpus_fingerprint(docs)
    if "--force" not in sys.argv and idx.load(out, fingerprint=fp):
        print(f"index in {out} is up to date with the docs (fingerprint {fp}); nothing to do. Use --force to rebuild.")
        sys.exit(0)
    idx.build(load_corpus(docs))
    idx.save(out, fingerprint=fp)
    print(f"indexed {len(idx.chunks)} chunks from {len({c.source for c in idx.chunks})} files "
          f"with {idx.embedder.name} in {time.perf_counter() - t0:.1f}s -> {out}")
