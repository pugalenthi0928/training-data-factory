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
    args = ap.parse_args()

    rows = load_jsonl(Path(args.input))
    if not rows:
        raise SystemExit("No rows to check.")

    checker = ContaminationChecker()

    # Load benchmark files
    for bp in args.benchmark:
        bp_path = Path(bp)
        if not bp_path.exists():
            print(f"Warning: benchmark file not found: {bp}", file=sys.stderr)
            continue
        checker.load_benchmark_file(bp_path, name=bp_path.stem)

    # Load plain-text benchmark
    if args.benchmark_texts:
        tp = Path(args.benchmark_texts)
        if tp.exists():
            texts = [line.strip() for line in tp.read_text(encoding="utf-8").splitlines() if line.strip()]
            checker.load_custom_texts(texts, name=tp.stem)

    if checker.index.size == 0:
        print("Warning: no benchmark data loaded. Provide --benchmark or --benchmark-texts.", file=sys.stderr)
        print(json.dumps({"error": "no_benchmark_data", "total_examples": len(rows)}, indent=2))
        return

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


if __name__ == "__main__":
    main()
