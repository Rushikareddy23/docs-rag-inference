"""Thin async client for any OpenAI-compatible chat endpoint (vLLM, Triton's OpenAI frontend, ...)."""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict)


class LLMClient:
    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: float = 120.0):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
        # vLLM supports OpenAI-style structured outputs; turn off for servers that don't.
        self.structured = os.getenv("STRUCTURED_OUTPUT", "json_schema")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def chat(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.1,
        json_schema: dict | None = None,
    ) -> LLMResult:
        body: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_schema and self.structured == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "agent_step", "schema": json_schema},
            }
        t0 = time.perf_counter()
        r = await self._client.post(f"{self.base_url}/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage") or {}
        return LLMResult(
            text=data["choices"][0]["message"]["content"] or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=(time.perf_counter() - t0) * 1000,
            raw=data,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def parse_json_loose(text) -> dict | None:
    """Return the JSON *object* in a model reply, or None.

    Small models sometimes wrap JSON in prose or code fences, so the first {...} span is tried
    too. Anything that is not an object (a list like [1], a bare true, a number, a string) is
    rejected: callers can rely on getting a dict or None, never another type."""
    if not isinstance(text, str):
        return None
    try:
        value = json.loads(text)
        # valid JSON: accept it only if it is an object; don't dig objects out of lists
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    # not valid JSON as a whole (prose, code fences): look for an embedded object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            value = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None
    return None


def str_field(obj: dict, key: str) -> str:
    """A stripped string field, or "" when it is missing or not a string (e.g. "query": 5)."""
    value = obj.get(key)
    return value.strip() if isinstance(value, str) else ""
