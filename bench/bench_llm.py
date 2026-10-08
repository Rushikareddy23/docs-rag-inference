"""Benchmark an OpenAI-compatible LLM server (vLLM) under concurrent load.

Measures, per concurrency level:
  - TTFT (time to first token) p50 / p95   -> how fast users see a response start
  - end-to-end latency p50 / p95
  - output throughput (tokens/s across all requests) -> what continuous batching buys you

Usage:
  python bench/bench_llm.py --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-0.5B-Instruct \
      --concurrency 1 4 8 16 --requests 32 --max-tokens 128 --out results/llm_cpu.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

PROMPTS = [
    "Explain dynamic batching in an inference server in three sentences.",
    "What is the KV cache in a transformer and why does it use GPU memory?",
    "Compare tensor parallelism and pipeline parallelism briefly.",
    "Give three tips to reduce latency when serving large language models.",
    "What does PagedAttention improve compared with naive KV cache allocation?",
    "Describe what a model repository is in an inference server.",
]


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


async def one_request(client: httpx.AsyncClient, url: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft, tokens, chunks = None, 0, 0
    async with client.stream("POST", f"{url}/chat/completions", json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            if data.get("usage"):
                tokens = data["usage"].get("completion_tokens", tokens)
            for ch in data.get("choices", []):
                if (ch.get("delta") or {}).get("content"):
                    chunks += 1
                    if ttft is None:
                        ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return {"ttft": ttft if ttft is not None else total, "latency": total, "tokens": tokens or chunks}


async def run_level(url: str, model: str, concurrency: int, n: int, max_tokens: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    errors = 0

    async with httpx.AsyncClient(timeout=600) as client:

        async def worker(i: int):
            nonlocal errors
            async with sem:
                try:
                    results.append(await one_request(client, url, model, PROMPTS[i % len(PROMPTS)], max_tokens))
                except httpx.HTTPError:
                    errors += 1

        # warm-up so the first measured request doesn't pay for graph compilation
        await one_request(client, url, model, PROMPTS[0], 8)
        t0 = time.perf_counter()
        await asyncio.gather(*(worker(i) for i in range(n)))
        wall = time.perf_counter() - t0

    toks = sum(r["tokens"] for r in results)
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "errors": errors,
        "ttft_p50_ms": round(pct([r["ttft"] for r in results], 0.50) * 1000, 1),
        "ttft_p95_ms": round(pct([r["ttft"] for r in results], 0.95) * 1000, 1),
        "latency_p50_ms": round(pct([r["latency"] for r in results], 0.50) * 1000, 1),
        "latency_p95_ms": round(pct([r["latency"] for r in results], 0.95) * 1000, 1),
        "output_tokens_per_s": round(toks / wall, 1) if wall else 0,
        "per_request_tokens_per_s": round(
            statistics.mean(r["tokens"] / max(r["latency"] - r["ttft"], 1e-6) for r in results), 1
        )
        if results
        else 0,
        "wall_s": round(wall, 2),
    }


def table(rows: list[dict]) -> str:
    head = "| concurrency | TTFT p50 | TTFT p95 | latency p50 | latency p95 | throughput (tok/s) | errors |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [
        f"| {r['concurrency']} | {r['ttft_p50_ms']} ms | {r['ttft_p95_ms']} ms | {r['latency_p50_ms']} ms | "
        f"{r['latency_p95_ms']} ms | {r['output_tokens_per_s']} | {r['errors']} |"
        for r in rows
    ]
    return "\n".join([head, sep, *lines])


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--label", default="", help="e.g. 'cpu-laptop' or 'L4 GPU'")
    ap.add_argument("--out", default="results/llm.json")
    a = ap.parse_args()

    rows = []
    for c in a.concurrency:
        print(f"concurrency {c} ...", flush=True)
        rows.append(await run_level(a.base_url.rstrip("/"), a.model, c, a.requests, a.max_tokens))
    report = {"label": a.label, "model": a.model, "max_tokens": a.max_tokens, "results": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    md = f"## {a.label or 'LLM benchmark'} — {a.model}\n\n{table(rows)}\n"
    Path(a.out).with_suffix(".md").write_text(md)
    print("\n" + md)
    if len(rows) > 1 and rows[0]["output_tokens_per_s"]:
        print(
            f"Throughput gain from batching: {rows[-1]['output_tokens_per_s'] / rows[0]['output_tokens_per_s']:.1f}x "
            f"(concurrency {rows[0]['concurrency']} -> {rows[-1]['concurrency']})"
        )


if __name__ == "__main__":
    asyncio.run(main())
