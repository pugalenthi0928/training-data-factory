#!/usr/bin/env python3
"""CLI wrapper for contamination detection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from training_data_robo.contamination import ContaminationChecker
from training_data_robo.io import load_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Check training data for benchmark contamination.")
    ap.add_argument("--input", required=True, help="Input JSONL dataset to check")
    ap.add_argument("--benchmark", action="append", default=[], help="Path to benchmark JSONL file(s)")
    ap.add_argument("--benchmark-texts", default=None, help="Plain text file (one entry per line) to use as benchmark")
    ap.add_argument("--output", default=None, help="Write contamination report JSON here")
    ap.add_argument("--text-fields", default="output_text,input_text", help="Comma-separated fields to check")
    ap.add_argument(
        "--fail-on-contamination",
        action="store_true",
        help="Exit nonzero when any overlap crosses the contamination threshold",
    )
    args = ap.parse_args()

    if not args.benchmark and not args.benchmark_texts:
        ap.error("provide at least one --benchmark or --benchmark-texts input")

    rows = load_jsonl(Path(args.input))
    if not rows:
        raise SystemExit("No rows to check.")

    checker = ContaminationChecker()

    # Load benchmark files
    for bp in args.benchmark:
        bp_path = Path(bp)
        if not bp_path.exists():
            raise SystemExit(f"Benchmark file not found: {bp}")
        checker.load_benchmark_file(bp_path, name=bp_path.stem)

    # Load plain-text benchmark
    if args.benchmark_texts:
        tp = Path(args.benchmark_texts)
        if not tp.exists():
            raise SystemExit(f"Benchmark text file not found: {tp}")
        texts = [line.strip() for line in tp.read_text(encoding="utf-8").splitlines() if line.strip()]
        checker.load_custom_texts(texts, name=tp.stem)

    if checker.index.size == 0:
        raise SystemExit("No usable benchmark text was loaded")

    text_fields = [f.strip() for f in args.text_fields.split(",")]
    report = checker.check_dataset(rows, text_fields=text_fields)

    # Print summary (without per-example details)
    summary = {k: v for k, v in report.items() if k != "per_example"}
    print(json.dumps(summary, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nFull report written to {out_path}")

    if args.fail_on_contamination and report["contaminated_count"] > 0:
        raise SystemExit(f"Contamination gate failed: {report['contaminated_count']} examples flagged")


if __name__ == "__main__":
    main()
