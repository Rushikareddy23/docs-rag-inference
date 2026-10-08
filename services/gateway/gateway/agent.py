"""Two answer strategies over the same retriever + LLM:

- rag:   retrieve once, answer once. 1 LLM call, lowest latency.
- agent: the model decides each step whether to search (and with what query) or answer.
         Before a draft answer is returned, a separate evidence check asks whether the
         cited sources actually support it; if not, the agent searches for what is missing.
         Handles vague or multi-part questions better, costs extra LLM calls.

Comparing the two in bench/ is a good interview talking point: quality vs. latency.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import httpx

from .llm import LLMClient, parse_json_loose, str_field

STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["search", "answer"]},
        "query": {"type": "string"},
        "answer": {"type": "string"},
    },
    "required": ["action"],
}

AGENT_SYSTEM = """You answer questions about LLM inference software (vLLM, NVIDIA Triton) using ONLY the documentation excerpts you retrieve.
Each turn, reply with a JSON object and nothing else:
  {"action": "search", "query": "<short search query>"}   to look something up
  {"action": "answer", "answer": "<2-4 sentences, each ending with its source like [S1]>"}   when the excerpts are enough
Search at least once before answering. Use only facts stated in the excerpts, never outside knowledge.
If the excerpts do not contain the answer, say so instead of guessing."""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "missing": {"type": "string"},
    },
    "required": ["supported"],
}

VERIFY_SYSTEM = """You check answers against documentation. Given excerpts and a draft answer, decide whether
every claim in the draft is directly supported by the excerpts. Reply with JSON only:
  {"supported": true}
  {"supported": false, "missing": "<short search query for the unsupported part>"}"""

RAG_SYSTEM = """You answer questions about LLM inference software (vLLM, NVIDIA Triton) using ONLY the documentation excerpts below.
Answer in 2-4 sentences. End every sentence with the source it came from, like [S1] or [S2].
Use only facts stated in the excerpts, never outside knowledge. If the excerpts do not contain the answer, say you don't know."""


@dataclass
class Trace:
    steps: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_ms: float = 0.0
    retrieval_ms: float = 0.0


class Retriever:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def search(self, query: str, k: int) -> list[dict]:
        r = await self._client.post(f"{self.base_url}/search", json={"query": query, "k": k})
        r.raise_for_status()
        return r.json()["hits"]

    async def aclose(self) -> None:
        await self._client.aclose()


def _format_hits(hits: list[dict], start: int) -> str:
    return "\n\n".join(
        f"[S{start + i}] ({h['source']} — {h['section'] or 'intro'})\n{h['text']}" for i, h in enumerate(hits)
    )


_WORD = re.compile(r"[a-z0-9_\-]{3,}")
_STOP = set("the and for that this with from are was were will can into have has its not you your use used using when which what how does also more than then they them their there been being".split())
_CITE = re.compile(r"\[S\d+\]")


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def _matching_close(t: str) -> int:
    """Index of the "]" that closes the "[" at t[0], respecting nesting; -1 if unbalanced."""
    depth = 0
    for i, ch in enumerate(t):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _is_json_structure(t: str) -> bool:
    try:
        return isinstance(json.loads(t), (list, dict))
    except (json.JSONDecodeError, ValueError):
        return False


def clean_answer(text) -> str:
    """Undo the one wrapper small models were observed adding: square brackets around the
    whole answer, e.g. "[Triton batches requests [S2]. It raises throughput [S3].]".

    Exactly two transformations, nothing else:
      1. "[...]" is removed only if the text is not valid JSON, it is not a citation like [S1],
         the "[" at the start is closed by the "]" at the very end (nesting respected), and the
         inside contains whitespace. So ["hello", "world"], [1, 2], [TODO] and
         "[link](url) ..." are kept.
      2. Matching quotes around the whole answer are removed if that quote appears nowhere inside.
    Angle brackets, parentheses and braces are never touched, so HTML such as
    <img src="diagram.png">, (asides) and {objects} pass through unchanged."""
    t = text.strip() if isinstance(text, str) else ""
    for _ in range(2):  # at most two layers, e.g. a quoted bracketed answer
        if len(t) < 3 or _is_json_structure(t):
            break
        inner = t[1:-1]
        if t[0] == "[" and not _CITE.match(t) and _matching_close(t) == len(t) - 1 and re.search(r"\s", inner.strip()):
            t = inner.strip()
        elif t[0] in "\"'" and t[-1] == t[0] and t[0] not in inner:
            t = inner.strip()
        else:
            break
    return t


def attribute(answer: str, sources: list[dict], min_overlap: float = 0.5) -> str:
    """Post-hoc attribution: give each uncited sentence the [S#] of the retrieved source it
    overlaps with most. Deterministic and free (no LLM call), so citations still appear when a
    small model ignores the instruction. Sentences with weak overlap stay uncited, which is an
    honest signal that the claim may not come from the docs."""
    if not sources or not answer:
        return answer
    src_terms = [_terms(s["text"]) for s in sources]
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", answer.strip()):
        words = _terms(sent)
        if _CITE.search(sent) or len(words) < 3:
            out.append(sent)
            continue
        scores = [len(words & st) / len(words) for st in src_terms]
        best = max(range(len(scores)), key=scores.__getitem__)
        if scores[best] >= min_overlap:
            tag = f" [S{best + 1}]"
            sent = sent[:-1] + tag + sent[-1] if sent[-1] in ".!?" else sent + tag
        out.append(sent)
    return " ".join(out)


class Answerer:
    def __init__(self, llm: LLMClient, retriever: Retriever, k: int = 4, max_steps: int = 3, verify: bool = True):
        self.llm, self.retriever, self.k, self.max_steps, self.verify = llm, retriever, k, max_steps, verify

    async def _check_evidence(self, question: str, draft: str, trace: Trace) -> dict:
        """Ask the model (in a fresh context) whether the sources support the draft."""
        res = await self._call(
            [
                {"role": "system", "content": VERIFY_SYSTEM},
                {
                    "role": "user",
                    "content": f"Documentation:\n{_format_hits(trace.sources, 1)}\n\n"
                    f"Question: {question}\n\nDraft answer: {draft}",
                },
            ],
            trace,
            max_tokens=128,
            temperature=0.0,
            json_schema=VERIFY_SCHEMA,
        )
        verdict = parse_json_loose(res.text) or {}
        raw = verdict.get("supported")
        # Only a real JSON boolean counts. Anything else (unparseable reply, "yes", 1, missing)
        # is "unverified" (None): it is never reported as success, and it does not trigger
        # another search, so a flaky checker can't loop or burn the step budget.
        supported = raw if isinstance(raw, bool) else None
        out = {"supported": supported, "missing": str_field(verdict, "missing")}
        if supported is None:
            out["error"] = "unreadable verdict"
        trace.steps.append({"action": "verify", **out})
        return out

    async def _retrieve(self, query: str, trace: Trace) -> str:
        t0 = time.perf_counter()
        hits = [h for h in await self.retriever.search(query, self.k) if h["id"] not in {s["id"] for s in trace.sources}]
        trace.retrieval_ms += (time.perf_counter() - t0) * 1000
        block = _format_hits(hits, len(trace.sources) + 1) if hits else "(no new results)"
        trace.sources.extend(hits)
        trace.steps.append({"action": "search", "query": query, "results": len(hits)})
        return block

    async def _call(self, messages: list[dict], trace: Trace, **kw):
        res = await self.llm.chat(messages, **kw)
        trace.llm_calls += 1
        trace.prompt_tokens += res.prompt_tokens
        trace.completion_tokens += res.completion_tokens
        trace.llm_ms += res.latency_ms
        return res

    async def rag(self, question: str) -> tuple[str, Trace]:
        trace = Trace()
        context = await self._retrieve(question, trace)
        res = await self._call(
            [
                {"role": "system", "content": RAG_SYSTEM},
                {"role": "user", "content": f"Documentation:\n{context}\n\nQuestion: {question}"},
            ],
            trace,
            max_tokens=256,
        )
        trace.steps.append({"action": "answer"})
        return attribute(clean_answer(res.text), trace.sources), trace

    async def agent(self, question: str) -> tuple[str, Trace]:
        trace = Trace()
        messages = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": question}]
        for step_no in range(self.max_steps):
            res = await self._call(messages, trace, max_tokens=384, json_schema=STEP_SCHEMA)
            step = parse_json_loose(res.text) or {}  # always a dict, even for "[1]" or "true"
            action = step.get("action")
            messages.append({"role": "assistant", "content": res.text})
            draft = str_field(step, "answer")
            if action == "answer" and trace.sources and draft:
                if not self.verify:
                    trace.steps.append({"action": "answer"})
                    return attribute(clean_answer(draft), trace.sources), trace
                verdict = await self._check_evidence(question, draft, trace)
                # stop when supported, when the verdict was unreadable (None), or out of steps;
                # "verified" carries the honest outcome: True, False, or None (unverified)
                if verdict["supported"] is not False or step_no == self.max_steps - 1:
                    trace.steps.append({"action": "answer", "verified": verdict["supported"]})
                    return attribute(clean_answer(draft), trace.sources), trace
                # not supported: look up the missing piece and let the agent revise
                query = verdict["missing"] or question
                observation = await self._retrieve(query, trace)
                messages.append(
                    {
                        "role": "user",
                        "content": "Your draft was not fully supported by the sources. "
                        f"Search results for '{query}':\n{observation}\nRevise your answer.",
                    }
                )
                continue
            # anything else (search, malformed output, answering before searching) -> search
            query = str_field(step, "query") or question
            observation = await self._retrieve(query, trace)
            messages.append({"role": "user", "content": f"Search results for '{query}':\n{observation}"})
        # step budget used up: force a grounded answer from everything gathered so far
        context = _format_hits(trace.sources, 1)
        res = await self._call(
            [
                {"role": "system", "content": RAG_SYSTEM},
                {"role": "user", "content": f"Documentation:\n{context}\n\nQuestion: {question}"},
            ],
            trace,
        )
        trace.steps.append({"action": "answer", "forced": True})
        return attribute(clean_answer(res.text), trace.sources), trace
