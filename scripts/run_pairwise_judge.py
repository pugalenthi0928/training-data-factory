#!/usr/bin/env python3
"""Run a versioned model judge over primary and swapped blind presentations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.pairwise_judge import PAIRWISE_JUDGE_PROMPT_SHA256, run_pairwise_judge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Forge pairwise judge")
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--judge-family", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    predictions = run_pairwise_judge(
        args.packet,
        args.output,
        model=args.model,
        judge_family=args.judge_family,
    )
    print(
        json.dumps(
            {
                "predictions": len(predictions),
                "model": args.model,
                "judge_family": args.judge_family,
                "prompt_sha256": PAIRWISE_JUDGE_PROMPT_SHA256,
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
