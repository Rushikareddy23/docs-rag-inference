"""Gateway microservice: the public API. Orchestrates retrieval + LLM calls and reports timings."""
from __future__ import annotations

import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import Answerer, Retriever
from .llm import LLMClient

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = LLMClient()
    retriever = Retriever(os.getenv("RETRIEVER_URL", "http://localhost:8001"))
    state["answerer"] = Answerer(
        llm, retriever, k=int(os.getenv("TOP_K", "4")), max_steps=int(os.getenv("MAX_AGENT_STEPS", "3")),
        verify=os.getenv("VERIFY_ANSWERS", "true").lower() == "true",
    )
    state["llm"], state["retriever"] = llm, retriever
    yield
    await llm.aclose()
    await retriever.aclose()


app = FastAPI(title="docs-rag gateway", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    mode: Literal["agent", "rag"] = "agent"


class Source(BaseModel):
    ref: str
    source: str
    section: str
    score: float


class AskResponse(BaseModel):
    answer: str
    mode: str
    sources: list[Source]
    steps: list[dict]
    usage: dict
    timings_ms: dict


@app.post("/v1/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    answerer: Answerer = state["answerer"]
    t0 = time.perf_counter()
    try:
        answer, trace = await (answerer.agent(req.question) if req.mode == "agent" else answerer.rag(req.question))
    except httpx.HTTPError as exc:
        # upstream (vLLM or retriever) failed: return 502 instead of a stack trace
        raise HTTPException(status_code=502, detail=f"upstream error: {exc.__class__.__name__}: {exc}") from exc
    total = (time.perf_counter() - t0) * 1000
    if not answer.strip():
        answer = "The model returned an empty answer. Try again, or switch to the other mode."
    return AskResponse(
        answer=answer,
        mode=req.mode,
        sources=[
            Source(ref=f"S{i + 1}", source=h["source"], section=h["section"], score=h["score"])
            for i, h in enumerate(trace.sources)
        ],
        steps=trace.steps,
        usage={
            "citations": len(set(re.findall(r"\[S\d+\]", answer))),
            "llm_calls": trace.llm_calls,
            "prompt_tokens": trace.prompt_tokens,
            "completion_tokens": trace.completion_tokens,
        },
        timings_ms={
            "total": round(total, 1),
            "llm": round(trace.llm_ms, 1),
            "retrieval": round(trace.retrieval_ms, 1),
        },
    )


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "model": state["llm"].model}


@app.get("/readyz", include_in_schema=False)
async def readyz() -> dict:
    """Ready only when both dependencies answer: Kubernetes won't route traffic before that."""
    llm: LLMClient = state["llm"]
    retriever: Retriever = state["retriever"]
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            (await c.get(f"{llm.base_url}/models")).raise_for_status()
            (await c.get(f"{retriever.base_url}/healthz")).raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"ready": True}


# Serve the built React UI (services/gateway/web/dist) at "/" when it exists.
# Mounted last so the API routes above take precedence.
_WEB_DIST = Path(os.getenv("WEB_DIST", Path(__file__).resolve().parents[1] / "web" / "dist"))
if _WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIST, html=True), name="web")
