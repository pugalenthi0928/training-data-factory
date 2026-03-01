#!/usr/bin/env python3
"""Split a JSONL dataset into train/test sets with stratification by task type."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from training_data_robo.io import load_jsonl, write_jsonl


def stratified_split(
    rows: List[Dict[str, Any]],
    test_fraction: float = 0.2,
    stratify_field: str = "task_name",
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split rows into train/test with proportional representation per group."""
    rng = random.Random(seed)

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(stratify_field, "unknown"))
        groups.setdefault(key, []).append(row)

    train: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    for key in sorted(groups.keys()):
        group = groups[key]
        rng.shuffle(group)
        n_test = max(1, int(len(group) * test_fraction))
        if len(group) <= 2:
            # Too few examples — put all in train
            train.extend(group)
        else:
            test.extend(group[:n_test])
            train.extend(group[n_test:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def main() -> None:
    ap = argparse.ArgumentParser(description="Split JSONL dataset into train/test sets.")
    ap.add_argument("--input", required=True, help="Input JSONL dataset")
    ap.add_argument("--train-output", required=True, help="Output path for training set")
    ap.add_argument("--test-output", required=True, help="Output path for test set")
    ap.add_argument("--test-fraction", type=float, default=0.2, help="Fraction for test set (default: 0.2)")
    ap.add_argument("--stratify-field", default="task_name", help="Field to stratify by (default: task_name)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    rows = load_jsonl(Path(args.input))
    if not rows:
        raise SystemExit("No rows to split.")

    train, test = stratified_split(
        rows,
        test_fraction=args.test_fraction,
        stratify_field=args.stratify_field,
        seed=args.seed,
    )

    write_jsonl(Path(args.train_output), train)
    write_jsonl(Path(args.test_output), test)

    # Summary
    train_dist = Counter(str(r.get(args.stratify_field, "unknown")) for r in train)
    test_dist = Counter(str(r.get(args.stratify_field, "unknown")) for r in test)
    summary = {
        "total": len(rows),
        "train": len(train),
        "test": len(test),
        "train_distribution": dict(train_dist),
        "test_distribution": dict(test_dist),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
