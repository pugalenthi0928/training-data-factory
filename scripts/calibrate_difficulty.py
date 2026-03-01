#!/usr/bin/env python3
"""CLI wrapper for difficulty calibration."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from training_data_robo.difficulty import calibrate_batch
from training_data_robo.io import load_jsonl, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Tag training examples with difficulty levels.")
    ap.add_argument("--input", required=True, help="Input JSONL dataset")
    ap.add_argument("--output", required=True, help="Output JSONL with difficulty field")
    args = ap.parse_args()

    rows = load_jsonl(Path(args.input))
    if not rows:
        raise SystemExit("No rows to calibrate.")

    summary = calibrate_batch(rows)
    write_jsonl(Path(args.output), rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
