#!/usr/bin/env python3
"""Measure curation controls on a labelled, controlled pair fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.calibration import evaluate_calibration, load_calibration_pairs
from forge.similarity import build_encoder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Forge dedupe controls on labelled pair fixtures")
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--semantic-backend",
        choices=("disabled", "sentence_transformers", "openai"),
        default="disabled",
    )
    parser.add_argument("--semantic-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--semantic-revision", default="main")
    parser.add_argument("--min-fuzzy-precision", type=float, default=0.0)
    parser.add_argument("--min-fuzzy-recall", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    encoder = build_encoder(args.semantic_backend, args.semantic_model, args.semantic_revision)
    report = evaluate_calibration(load_calibration_pairs(args.fixture), encoder=encoder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    selected = report["fuzzy_minhash_lsh"]["selected"]
    if float(selected["precision"]) < args.min_fuzzy_precision:
        raise SystemExit(f"fuzzy precision {selected['precision']} is below {args.min_fuzzy_precision}")
    if float(selected["recall"]) < args.min_fuzzy_recall:
        raise SystemExit(f"fuzzy recall {selected['recall']} is below {args.min_fuzzy_recall}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
