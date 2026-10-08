import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "retriever"))

from retriever.chunking import chunk_markdown, corpus_fingerprint, load_corpus  # noqa: E402
from retriever.embeddings import HashEmbedder  # noqa: E402
from retriever.index import BM25, VectorIndex, tokenize  # noqa: E402

DOC = """# Dynamic Batcher
Dynamic batching lets the server combine individual inference requests into a larger batch.
Set max_queue_delay_microseconds to control how long requests wait.

# Instance Groups
Use instance_group to run several copies of a model on one GPU.
"""


def test_chunks_keep_their_heading():
    chunks = chunk_markdown(DOC, "triton/batcher.md")
    assert [c.section for c in chunks] == ["Dynamic Batcher", "Instance Groups"]
    assert chunks[0].text.startswith("Dynamic Batcher")
    assert all(c.source == "triton/batcher.md" for c in chunks)


def test_long_sections_are_windowed_with_overlap():
    body = "# Big\n" + " ".join(f"Sentence number {i} about batching." for i in range(300))
    chunks = chunk_markdown(body, "big.md", size=400, overlap=80)
    assert len(chunks) > 5
    assert all(len(c.text) <= 400 + len("Big\n") + 2 for c in chunks)
    # consecutive windows share text (overlap) so facts on a boundary aren't lost
    assert chunks[0].text[-40:].split()[-1] in chunks[1].text


def test_search_ranks_the_relevant_chunk_first(tmp_path):
    idx = VectorIndex(HashEmbedder())
    idx.build(chunk_markdown(DOC, "triton/batcher.md"))
    top, score = idx.search("how do I run copies of a model with instance_group", k=1)[0]
    assert top.section == "Instance Groups" and score > 0
    # save / load round trip
    idx.save(tmp_path)
    again = VectorIndex(HashEmbedder())
    assert again.load(tmp_path) and len(again.chunks) == len(idx.chunks)


def test_load_corpus_walks_markdown_files(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.md").write_text(DOC)
    (tmp_path / "skip.txt").write_text("not markdown")
    chunks = load_corpus(tmp_path)
    assert {c.source for c in chunks} == {"a/one.md"}


def test_tokenizer_keeps_and_splits_compound_terms():
    toks = tokenize("Set gpu-memory-utilization in config.pbtxt")
    assert {"gpu-memory-utilization", "config.pbtxt", "gpu", "memory", "pbtxt"} <= set(toks)


def test_bm25_finds_exact_terms():
    docs = ["the model configuration lives in config.pbtxt", "dynamic batching groups requests", "kv cache memory"]
    scores = BM25(docs).scores("which file is config.pbtxt")
    assert scores.argmax() == 0 and scores[1] == 0


def test_hybrid_mode_surfaces_keyword_matches(tmp_path):
    idx = VectorIndex(HashEmbedder())
    idx.build(chunk_markdown(DOC, "triton/batcher.md"))
    for mode in ("dense", "bm25", "hybrid"):
        top, _ = idx.search("instance_group", k=1, mode=mode)[0]
        assert top.section == "Instance Groups", mode
    # BM25 is rebuilt when an index is loaded from disk
    idx.save(tmp_path)
    again = VectorIndex(HashEmbedder())
    assert again.load(tmp_path) and again.bm25 is not None


def test_saved_index_is_rejected_when_docs_change(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text(DOC)
    fp = corpus_fingerprint(docs)
    idx = VectorIndex(HashEmbedder())
    idx.build(load_corpus(docs))
    idx.save(tmp_path / "index", fingerprint=fp)
    assert VectorIndex(HashEmbedder()).load(tmp_path / "index", fingerprint=fp)
    (docs / "a.md").write_text(DOC + "\n# New section\nnew text")
    assert corpus_fingerprint(docs) != fp
    assert not VectorIndex(HashEmbedder()).load(tmp_path / "index", fingerprint=corpus_fingerprint(docs))
