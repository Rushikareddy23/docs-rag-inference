"""Measure retrieval quality: does the right documentation show up in the top-k results?

A question counts as a hit@k if any of its expected key phrases appears in the top-k chunks.
Run against the retriever service:
  python bench/eval_retrieval.py --url http://localhost:8001 --k 1 4 --out results/retrieval.json
Add your own questions to bench/questions.jsonl to make the number more meaningful.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


def is_hit(hits: list[dict], expected: list[str]) -> bool:
    blob = " ".join((h["text"] + " " + h["source"]).lower() for h in hits)
    return any(e.lower() in blob for e in expected)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001")
    ap.add_argument("--questions", default=str(Path(__file__).with_name("questions.jsonl")))
    ap.add_argument("--k", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--mode", nargs="+", default=["hybrid"], choices=["hybrid", "dense", "bm25"])
    ap.add_argument("--out", default="results/retrieval.json")
    a = ap.parse_args()

    qs = [json.loads(line) for line in Path(a.questions).read_text().splitlines() if line.strip()]
    kmax = max(a.k)
    report = {}
    with httpx.Client(timeout=30) as c:
        for mode in a.mode:
            hits_at = {k: 0 for k in a.k}
            rows, search_ms = [], []
            for q in qs:
                r = c.post(f"{a.url.rstrip('/')}/search", json={"query": q["question"], "k": kmax, "mode": mode})
                r.raise_for_status()
                data = r.json()
                search_ms.append(data["took_ms"])
                row = {"question": q["question"]}
                for k in a.k:
                    ok = is_hit(data["hits"][:k], q["expected"])
                    hits_at[k] += ok
                    row[f"hit@{k}"] = ok
                row["top_source"] = data["hits"][0]["source"] if data["hits"] else None
                rows.append(row)
            summary = {f"hit@{k}": round(hits_at[k] / len(qs), 3) for k in a.k}
            summary["questions"] = len(qs)
            summary["avg_search_ms"] = round(sum(search_ms) / len(search_ms), 2)
            report[mode] = {"summary": summary, "rows": rows}
            print(f"\n== {mode}")
            for row in rows:
                marks = " ".join(f"@{k}:{'✓' if row[f'hit@{k}'] else '✗'}" for k in a.k)
                print(f"{marks}  {row['question']}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print("\n| mode | " + " | ".join(f"hit@{k}" for k in a.k) + " | avg search |")
    print("|---|" + "---|" * (len(a.k) + 1))
    for mode, rep in report.items():
        sm = rep["summary"]
        print(f"| {mode} | " + " | ".join(f"{sm[f'hit@{k}']:.0%}" for k in a.k) + f" | {sm['avg_search_ms']} ms |")

if __name__ == "__main__":
    main()
