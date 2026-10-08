"""A tiny stand-in for vLLM's OpenAI-compatible API, used by tests and smoke runs.

It behaves like a well-behaved agent: first asks to search, then answers citing [S1].
Run standalone:  uvicorn tests.fake_llm:app --port 8000
"""
from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()


def _reply(messages: list[dict], wants_json: bool) -> str:
    if messages[0]["content"].startswith("You check answers"):
        return json.dumps({"supported": True})
    searched = any(m["role"] == "user" and m["content"].startswith("Search results") for m in messages)
    if not wants_json:
        return "Dynamic batching groups requests on the server to raise throughput [S1]."
    if not searched:
        return json.dumps({"action": "search", "query": messages[-1]["content"][:80]})
    return json.dumps({"action": "answer", "answer": "Triton's dynamic batcher combines requests into batches [S1]."})


@app.get("/v1/models")
async def models():
    return {"data": [{"id": "fake"}]}


@app.get("/health")
async def health():
    return {}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    text = _reply(body["messages"], "response_format" in body)
    words = text.split(" ")
    if not body.get("stream"):
        return {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": len(words)},
        }

    async def gen():
        for w in words:
            await asyncio.sleep(0.002)
            yield f"data: {json.dumps({'choices': [{'delta': {'content': w + ' '}}]})}\n\n"
        yield f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': len(words)}})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
