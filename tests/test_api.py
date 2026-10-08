"""HTTP-level tests for the gateway: malformed model output must never become a 500."""
import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "gateway"))

from gateway import main  # noqa: E402
from gateway.agent import Answerer  # noqa: E402
from gateway.llm import LLMResult  # noqa: E402

HITS = [{"id": "t#0", "source": "triton/batcher.md", "section": "Dynamic Batcher",
         "text": "The dynamic batcher combines requests into batches.", "score": 0.9}]


class FakeRetriever:
    async def search(self, query, k):
        return HITS


class LoopLLM:
    """Replies with the given strings in a loop, forever."""

    def __init__(self, replies):
        self.replies, self.i = replies, 0

    async def chat(self, messages, **kw):
        text = self.replies[self.i % len(self.replies)]
        self.i += 1
        return LLMResult(text=text, prompt_tokens=1, completion_tokens=1, latency_ms=0.1)


class DownLLM:
    async def chat(self, messages, **kw):
        raise httpx.ConnectError("connection refused")


def client_with(llm) -> TestClient:
    # no `with` block: skips the lifespan, so no real vLLM/retriever connections are made
    main.state["answerer"] = Answerer(llm, FakeRetriever(), k=2, max_steps=3)
    return TestClient(main.app)


GARBAGE = ["[1]", "true", "null", "5", '"text"', "not json", "{broken", "",
           json.dumps({"action": "search", "query": 5}),
           json.dumps({"action": "answer", "answer": ["x"]}),
           json.dumps({"action": 7}),
           json.dumps({"supported": "yes"})]


@pytest.mark.parametrize("mode", ["agent", "rag"])
@pytest.mark.parametrize("garbage", GARBAGE)
def test_malformed_model_output_returns_200(mode, garbage):
    r = client_with(LoopLLM([garbage])).post("/v1/ask", json={"question": "What is dynamic batching?", "mode": mode})
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["answer"], str) and body["answer"].strip()  # never an empty answer
    for step in body["steps"]:
        if step["action"] == "verify":
            assert step["supported"] in (True, False, None)


def test_unreadable_verdict_is_reported_as_unverified_over_http():
    llm = LoopLLM([json.dumps({"action": "search", "query": "batching"}),
                   json.dumps({"action": "answer", "answer": "It batches requests [S1]."}),
                   "not json"])
    body = client_with(llm).post("/v1/ask", json={"question": "What is dynamic batching?"}).json()
    assert body["steps"][-1] == {"action": "answer", "verified": None}


def test_upstream_failure_returns_502_not_500():
    r = client_with(DownLLM()).post("/v1/ask", json={"question": "What is dynamic batching?"})
    assert r.status_code == 502
    assert "upstream error" in r.json()["detail"]


def test_request_validation():
    c = client_with(LoopLLM(["x"]))
    assert c.post("/v1/ask", json={"question": "hi"}).status_code == 422          # too short
    assert c.post("/v1/ask", json={"question": "valid question", "mode": "x"}).status_code == 422
