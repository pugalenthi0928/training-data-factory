#!/usr/bin/env python3
"""CLI wrapper for LLM-as-Judge scoring."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from training_data_robo.io import load_jsonl, write_jsonl
from training_data_robo.judge import DummyJudge, LLMJudge


async def run(args: argparse.Namespace) -> None:
    rows = load_jsonl(Path(args.input))
    if not rows:
        raise SystemExit("No rows to judge.")

    if args.max_examples:
        rows = rows[: args.max_examples]

    if args.fake or not os.getenv("OPENAI_API_KEY"):
        judge = DummyJudge()
        print("[run_judge] Using DummyJudge (no API calls).", flush=True)
    else:
        judge = LLMJudge(
            model=args.model,
            max_concurrent=args.concurrency,
        )
        print(f"[run_judge] Using LLMJudge with {args.model}.", flush=True)

    results = await judge.judge_batch(rows)

    # Merge judge scores back into rows
    result_map = {r.example_id: r for r in results}
    for row in rows:
        rid = str(row.get("id", ""))
        if rid in result_map:
            jr = result_map[rid]
            row["judge_scores"] = jr.to_dict()["verdicts"]
            row["judge_avg_score"] = jr.avg_score

    out_path = Path(args.output)
    write_jsonl(out_path, rows)

    # Summary
    avg_scores = [r.avg_score for r in results]
    overall_avg = sum(avg_scores) / len(avg_scores) if avg_scores else 0.0
    summary = {
        "examples_judged": len(results),
        "overall_avg_score": round(overall_avg, 2),
        "output": str(out_path),
    }
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Score training examples with LLM-as-Judge.")
    ap.add_argument("--input", required=True, help="Input JSONL dataset")
    ap.add_argument("--output", required=True, help="Output JSONL with judge_scores field")
    ap.add_argument("--model", default="gpt-4.1-mini", help="Judge model (default: gpt-4.1-mini)")
    ap.add_argument("--concurrency", type=int, default=5, help="Max concurrent API calls")
    ap.add_argument("--max-examples", type=int, default=None, help="Limit examples to judge")
    ap.add_argument("--fake", action="store_true", help="Use DummyJudge (no API calls)")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
