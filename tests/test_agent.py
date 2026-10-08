import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "gateway"))

from gateway.agent import Answerer, attribute, clean_answer  # noqa: E402
from gateway.llm import LLMResult, parse_json_loose  # noqa: E402

HITS = [
    {"id": "t#0", "source": "triton/batcher.md", "section": "Dynamic Batcher", "text": "Dynamic batching...", "score": 0.9},
    {"id": "t#1", "source": "triton/model.md", "section": "Instances", "text": "instance_group...", "score": 0.5},
]


class FakeRetriever:
    def __init__(self):
        self.queries = []

    async def search(self, query, k):
        self.queries.append(query)
        return HITS[:k]


class ScriptedLLM:
    """Returns pre-scripted replies in order, recording what it was sent."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def chat(self, messages, **kw):
        self.calls.append((messages, kw))
        return LLMResult(text=self.replies.pop(0), prompt_tokens=10, completion_tokens=5, latency_ms=1.0)


def run(coro):
    return asyncio.run(coro)


def test_rag_mode_makes_one_llm_call_with_context():
    llm, ret = ScriptedLLM(["It batches requests [S1]."]), FakeRetriever()
    answer, trace = run(Answerer(llm, ret, k=2).rag("What is dynamic batching?"))
    assert answer == "It batches requests [S1]."
    assert trace.llm_calls == 1 and len(trace.sources) == 2
    assert "[S1] (triton/batcher.md" in llm.calls[0][0][1]["content"]


def test_agent_searches_then_answers():
    llm = ScriptedLLM(
        [json.dumps({"action": "search", "query": "dynamic batcher"}), json.dumps({"action": "answer", "answer": "Batches [S1]."}), json.dumps({"supported": True})]
    )
    ret = FakeRetriever()
    answer, trace = run(Answerer(llm, ret, k=2).agent("What is dynamic batching?"))
    assert answer == "Batches [S1]."
    assert ret.queries == ["dynamic batcher"]
    assert [s["action"] for s in trace.steps] == ["search", "verify", "answer"]
    assert trace.steps[-1]["verified"] is True
    assert llm.calls[0][1]["json_schema"]["properties"]["action"]["enum"] == ["search", "answer"]


def test_agent_refuses_to_answer_before_searching():
    llm = ScriptedLLM([json.dumps({"action": "answer", "answer": "guess"}), json.dumps({"action": "answer", "answer": "Grounded [S1]."}), json.dumps({"supported": True})])
    ret = FakeRetriever()
    answer, _ = run(Answerer(llm, ret).agent("What is dynamic batching?"))
    assert answer == "Grounded [S1]." and ret.queries == ["What is dynamic batching?"]


def test_agent_survives_garbage_output_and_forces_final_answer():
    llm = ScriptedLLM(["not json", "still not json", "{broken", "Final grounded answer [S1]."])
    ret = FakeRetriever()
    answer, trace = run(Answerer(llm, ret, k=2, max_steps=3).agent("q?"))
    assert answer == "Final grounded answer [S1]."
    assert trace.steps[-1] == {"action": "answer", "forced": True}
    # duplicate hits are not re-added across searches
    assert len(trace.sources) == 2


def test_parse_json_loose_handles_fenced_output():
    assert parse_json_loose('```json\n{"action": "search", "query": "x"}\n```') == {"action": "search", "query": "x"}
    assert parse_json_loose("nothing here") is None


def test_unsupported_draft_triggers_a_targeted_search():
    llm = ScriptedLLM(
        [
            json.dumps({"action": "search", "query": "dynamic batcher"}),
            json.dumps({"action": "answer", "answer": "It batches and also caches [S1]."}),
            json.dumps({"supported": False, "missing": "response cache"}),
            json.dumps({"action": "answer", "answer": "It batches [S1]; caching is separate [S2]."}),
            json.dumps({"supported": True}),
        ]
    )
    ret = FakeRetriever()
    answer, trace = run(Answerer(llm, ret, k=2, max_steps=4).agent("What does the dynamic batcher do?"))
    assert answer == "It batches [S1]; caching is separate [S2]."
    assert ret.queries == ["dynamic batcher", "response cache"]
    assert [s["action"] for s in trace.steps] == ["search", "verify", "search", "verify", "answer"]


def test_verification_can_be_turned_off():
    llm = ScriptedLLM([json.dumps({"action": "search", "query": "x"}), json.dumps({"action": "answer", "answer": "A [S1]."})])
    answer, trace = run(Answerer(llm, FakeRetriever(), verify=False).agent("q?"))
    assert answer == "A [S1]." and trace.llm_calls == 2


def test_attribution_adds_citations_only_where_sources_support_it():
    sources = [
        {"text": "The dynamic batcher delays requests in the scheduler so other requests can join the batch."},
        {"text": "gpu-memory-utilization sets the fraction of GPU memory vLLM may use for weights and KV cache."},
    ]
    out = attribute(
        "The scheduler delays requests so other requests can join the batch. "
        "vLLM uses a fraction of GPU memory for weights and KV cache. "
        "Bananas are yellow and delicious fruit.",
        sources,
    )
    assert "join the batch [S1]." in out
    assert "KV cache [S2]." in out
    assert out.endswith("Bananas are yellow and delicious fruit.")  # unsupported claim stays uncited


def test_attribution_keeps_existing_citations():
    assert attribute("Already cited [S2].", [{"text": "already cited here"}]) == "Already cited [S2]."


# --- regression tests for malformed model output (reported in review) -------------------------

import pytest  # noqa: E402

from gateway.llm import str_field  # noqa: E402


@pytest.mark.parametrize("reply", ["[1]", "true", "5", '"search"', "null", "[]", '[{"action": "answer"}]'])
def test_parse_json_loose_only_returns_objects(reply):
    assert parse_json_loose(reply) is None


def test_str_field_rejects_non_strings():
    assert str_field({"query": 5}, "query") == ""
    assert str_field({"query": ["a"]}, "query") == ""
    assert str_field({"query": "  ok "}, "query") == "ok"
    assert str_field({}, "query") == ""


def test_agent_survives_non_object_and_wrongly_typed_output():
    llm = ScriptedLLM(
        [
            "[1]",                                        # JSON, but a list
            "true",                                       # JSON, but a bool
            json.dumps({"action": "search", "query": 5}),  # numeric query
            "Final grounded answer [S1].",                # forced answer after the step budget
        ]
    )
    ret = FakeRetriever()
    answer, trace = run(Answerer(llm, ret, k=2, max_steps=3).agent("What is dynamic batching?"))
    assert answer == "Final grounded answer [S1]."
    assert ret.queries == ["What is dynamic batching?"] * 3  # bad queries fall back to the question
    assert trace.steps[-1] == {"action": "answer", "forced": True}


def test_non_string_answer_is_not_accepted_as_an_answer():
    llm = ScriptedLLM(
        [
            json.dumps({"action": "search", "query": "batching"}),
            json.dumps({"action": "answer", "answer": 7}),       # wrong type -> treated as search
            json.dumps({"action": "answer", "answer": "Batches [S1]."}),
            json.dumps({"supported": True}),
        ]
    )
    answer, trace = run(Answerer(llm, FakeRetriever(), max_steps=4).agent("q?"))
    assert answer == "Batches [S1]."
    assert [s["action"] for s in trace.steps] == ["search", "search", "verify", "answer"]


@pytest.mark.parametrize("verdict", ["not json", "[true]", "true", json.dumps({"supported": "yes"}), json.dumps({"supported": 1}), json.dumps({})])
def test_unreadable_verification_is_unverified_not_success(verdict):
    llm = ScriptedLLM([json.dumps({"action": "search", "query": "x"}), json.dumps({"action": "answer", "answer": "A [S1]."}), verdict])
    ret = FakeRetriever()
    answer, trace = run(Answerer(llm, ret, max_steps=3).agent("q?"))
    assert answer == "A [S1]."
    verify_step = next(s for s in trace.steps if s["action"] == "verify")
    assert verify_step["supported"] is None and verify_step["error"] == "unreadable verdict"
    assert trace.steps[-1] == {"action": "answer", "verified": None}
    assert ret.queries == ["x"]  # no extra search triggered by an unreadable verdict


def test_unsupported_on_last_step_is_reported_as_not_verified():
    llm = ScriptedLLM(
        [json.dumps({"action": "search", "query": "x"}), json.dumps({"action": "answer", "answer": "A [S1]."}), json.dumps({"supported": False, "missing": 3})]
    )
    _, trace = run(Answerer(llm, FakeRetriever(), max_steps=2).agent("q?"))
    assert trace.steps[-1] == {"action": "answer", "verified": False}
    assert next(s for s in trace.steps if s["action"] == "verify")["missing"] == ""  # non-string ignored


@pytest.mark.parametrize(
    "text, expected",
    [
        ('["hello", "world"] is a JSON array.', '["hello", "world"] is a JSON array.'),
        ("[S1] says batching raises throughput.", "[S1] says batching raises throughput."),
        ("[a] and [b] are both options.", "[a] and [b] are both options."),
        ("(See the docs) for details.", "(See the docs) for details."),
        ('"Fast" is relative, says "the doc".', '"Fast" is relative, says "the doc".'),
        ("[Batches requests [S2]. Raises throughput [S3].]", "Batches requests [S2]. Raises throughput [S3]."),
        ('"[Quoted and bracketed [S1].]"', "Quoted and bracketed [S1]."),
        ('"Quoted answer."', "Quoted answer."),
        ("[unbalanced answer", "[unbalanced answer"),
        ("[]", "[]"),
        # valid JSON is content, never a wrapper (reported in review)
        ('{"enabled": true}', '{"enabled": true}'),
        ('["hello", "world"]', '["hello", "world"]'),
        ("[1, 2, 3]", "[1, 2, 3]"),
        ('[{"a": [1]}]', '[{"a": [1]}]'),
        ('{"a": {"b": [1, 2]}}', '{"a": {"b": [1, 2]}}'),
        # braces and parentheses are never stripped
        ("{not json but braces}", "{not json but braces}"),
        ("(The whole answer in parentheses.)", "(The whole answer in parentheses.)"),
        # single tokens are kept
        ("[TODO]", "[TODO]"),
        # HTML and angle brackets are never touched (reported in review)
        ('<img src="diagram.png">', '<img src="diagram.png">'),
        ('<a href="https://docs.vllm.ai">vLLM docs</a>', '<a href="https://docs.vllm.ai">vLLM docs</a>'),
        ("<b>", "<b>"),
        ("<div>hello</div>", "<div>hello</div>"),
        ("<The whole answer in angle brackets [S1].>", "<The whole answer in angle brackets [S1].>"),
        ("[link text](https://example.com) explains it.", "[link text](https://example.com) explains it."),
        # still strips the real placeholder artifact, including around JSON-looking prose
        ("[Set enabled to true in the config [S1].]", "Set enabled to true in the config [S1]."),
        (None, ""),
        (42, ""),
    ],
)
def test_clean_answer_only_strips_whole_answer_wrappers(text, expected):
    assert clean_answer(text) == expected
