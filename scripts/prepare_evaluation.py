#!/usr/bin/env python3
"""Freeze an evaluation set and create blinded human and judge packets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.evaluation import freeze_evaluation_set


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a content-addressed Forge evaluation release")
    parser.add_argument("--items", type=Path, required=True, help="JSONL with Forge and baseline candidates")
    parser.add_argument("--protocol", type=Path, required=True, help="Frozen annotation protocol")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--author", required=True)
    parser.add_argument("--origin", required=True)
    parser.add_argument(
        "--independence-status",
        choices=("independent", "controlled_fixture"),
        required=True,
    )
    parser.add_argument("--generator-family", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-items", type=int, default=200)
    parser.add_argument("--minimum-overlap-items", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = freeze_evaluation_set(
        args.items,
        args.protocol,
        args.output_dir,
        author=args.author,
        origin=args.origin,
        independence_status=args.independence_status,
        generator_families=args.generator_family,
        seed=args.seed,
        target_items=args.target_items,
        minimum_overlap_items=args.minimum_overlap_items,
    )
    print(
        json.dumps(
            {
                "evaluation_id": manifest["evaluation_id"],
                "status": manifest["status"],
                "items": manifest["counts"]["items"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
