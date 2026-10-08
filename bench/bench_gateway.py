"""End-to-end benchmark of the gateway: compares the 'rag' and 'agent' answer modes.

  python bench/bench_gateway.py --url http://localhost:8080 --n 20 --out results/gateway.json

Reports p50/p95 latency, LLM calls per answer, and where the time goes (LLM vs retrieval).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx

from bench_llm import pct


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--questions", default=str(Path(__file__).with_name("questions.jsonl")))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="results/gateway.json")
    a = ap.parse_args()

    qs = [json.loads(x)["question"] for x in Path(a.questions).read_text().splitlines() if x.strip()][: a.n]
    report = {}
    with httpx.Client(timeout=600) as c:
        for mode in ("rag", "agent"):
            rows = []
            for q in qs:
                r = c.post(f"{a.url.rstrip('/')}/v1/ask", json={"question": q, "mode": mode})
                r.raise_for_status()
                rows.append(r.json())
            tot = [x["timings_ms"]["total"] for x in rows]
            llm = sum(x["timings_ms"]["llm"] for x in rows)
            ret = sum(x["timings_ms"]["retrieval"] for x in rows)
            report[mode] = {
                "questions": len(rows),
                "latency_p50_ms": round(pct(tot, 0.5), 1),
                "latency_p95_ms": round(pct(tot, 0.95), 1),
                "avg_llm_calls": round(sum(x["usage"]["llm_calls"] for x in rows) / len(rows), 2),
                "share_of_time_in_llm": round(llm / max(llm + ret, 1e-9), 3),
                "avg_retrieval_ms": round(ret / len(rows), 1),
            }
            print(mode, json.dumps(report[mode]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
