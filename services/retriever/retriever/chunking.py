"""Split Markdown documents into overlapping, heading-aware chunks.

Why heading-aware: a chunk that starts with its section title ("## Dynamic Batcher")
carries its own context, so both the embedder and the LLM understand it better.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

HEADING = re.compile(r"^(#{1,4})\s+(.*)$", re.MULTILINE)


@dataclass
class Chunk:
    id: str
    source: str  # relative file path, e.g. "triton/user_guide/model_configuration.md"
    section: str  # nearest heading above the chunk
    text: str


def _sections(markdown: str) -> list[tuple[str, str]]:
    """Return (heading, body) pairs. Text before the first heading gets heading ''."""
    out: list[tuple[str, str]] = []
    matches = list(HEADING.finditer(markdown))
    if not matches:
        return [("", markdown)]
    if matches[0].start() > 0:
        out.append(("", markdown[: matches[0].start()]))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        out.append((m.group(2).strip(), markdown[m.end() : end]))
    return out


def _windows(text: str, size: int, overlap: int) -> list[str]:
    """Cut text into ~size-character windows, breaking on paragraph or sentence ends."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # prefer to break at a paragraph, then a sentence, inside the last 30% of the window
            floor = start + int(size * 0.7)
            cut = max(text.rfind("\n\n", floor, end), text.rfind(". ", floor, end))
            if cut > floor:
                end = cut + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in pieces if p]


def chunk_markdown(markdown: str, source: str, size: int = 900, overlap: int = 150) -> list[Chunk]:
    # drop code-fence noise like HTML comments and image tags, keep code blocks (users ask about flags)
    markdown = re.sub(r"<!--.*?-->", "", markdown, flags=re.DOTALL)
    markdown = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", markdown)
    chunks: list[Chunk] = []
    for heading, body in _sections(markdown):
        for j, piece in enumerate(_windows(body, size, overlap)):
            text = f"{heading}\n{piece}" if heading else piece
            chunks.append(Chunk(id=f"{source}#{len(chunks)}", source=source, section=heading, text=text))
    return chunks


def load_corpus(docs_dir: str | Path, size: int = 900, overlap: int = 150) -> list[Chunk]:
    root = Path(docs_dir)
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*.md")):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        chunks.extend(chunk_markdown(content, str(path.relative_to(root)).replace("\\", "/"), size, overlap))
    return chunks


def corpus_fingerprint(docs_dir: str | Path) -> str:
    """Hash of every Markdown file's path and contents: changes whenever the docs change."""
    import hashlib

    h = hashlib.sha256()
    root = Path(docs_dir)
    for path in sorted(root.rglob("*.md")):
        h.update(str(path.relative_to(root)).replace("\\", "/").encode())
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()[:16]
