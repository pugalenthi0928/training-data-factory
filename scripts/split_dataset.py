#!/usr/bin/env python3
"""Split a JSONL dataset without leaking source documents across partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from training_data_robo.io import load_jsonl, write_jsonl


def stratified_split(
    rows: List[Dict[str, Any]],
    test_fraction: float = 0.2,
    stratify_field: str = "task_name",
    source_field: str = "document_id",
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split whole source groups into train and test partitions.

    The historical implementation shuffled examples within each task. That could
    place examples derived from the same document or chunk in both partitions.
    This implementation fails closed when provenance is missing and assigns every
    row from a source document to exactly one partition.

    ``stratify_field`` is retained for API compatibility and reporting. Source
    isolation takes priority over exact row-level task proportions.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")

    if not rows:
        return [], []

    rng = random.Random(seed)

    source_groups: Dict[str, List[Dict[str, Any]]] = {}
    missing_source_rows: List[int] = []
    for index, row in enumerate(rows):
        raw_source_id = row.get(source_field)
        source_id = str(raw_source_id).strip() if raw_source_id is not None else ""
        if not source_id:
            missing_source_rows.append(index)
            continue
        source_groups.setdefault(source_id, []).append(row)

    if missing_source_rows:
        preview = ", ".join(str(index) for index in missing_source_rows[:5])
        raise ValueError(
            f"{len(missing_source_rows)} rows are missing required provenance field "
            f"{source_field!r}; first row indexes: {preview}"
        )

    if len(source_groups) < 2:
        raise ValueError(f"At least two unique {source_field!r} values are required for a safe split")

    test_source_ids = _choose_test_sources(source_groups, test_fraction, rng)

    train: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    for source_id, group in source_groups.items():
        if source_id in test_source_ids:
            test.extend(group)
        else:
            train.extend(group)

    rng.shuffle(train)
    rng.shuffle(test)
    assert_no_identifier_overlap(train, test, source_field)
    assert_no_identifier_overlap(train, test, "chunk_id", allow_missing=True)
    return train, test


def _choose_test_sources(
    source_groups: Dict[str, List[Dict[str, Any]]],
    test_fraction: float,
    rng: random.Random,
) -> Set[str]:
    """Choose whole sources whose row count is close to the requested fraction."""
    source_ids = sorted(source_groups)
    rng.shuffle(source_ids)
    target_rows = sum(len(group) for group in source_groups.values()) * test_fraction
    target_sources = len(source_ids) * test_fraction

    candidates: List[Set[str]] = [{source_id} for source_id in source_ids]
    for _ in range(min(128, max(16, len(source_ids) * 4))):
        order = list(source_ids)
        rng.shuffle(order)
        chosen: Set[str] = set()
        chosen_rows = 0
        for source_id in order:
            if len(chosen) >= len(source_ids) - 1:
                break
            source_rows = len(source_groups[source_id])
            current_distance = abs(chosen_rows - target_rows)
            proposed_distance = abs(chosen_rows + source_rows - target_rows)
            if not chosen or proposed_distance < current_distance:
                chosen.add(source_id)
                chosen_rows += source_rows
        candidates.append(chosen)

    def score(candidate: Set[str]) -> Tuple[float, float]:
        row_count = sum(len(source_groups[source_id]) for source_id in candidate)
        return (
            abs(row_count - target_rows),
            abs(len(candidate) - target_sources),
        )

    return min(candidates, key=score)


def _identifier_values(
    rows: Iterable[Dict[str, Any]],
    field: str,
) -> Set[str]:
    return {str(row[field]).strip() for row in rows if row.get(field) is not None and str(row[field]).strip()}


def assert_no_identifier_overlap(
    train: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    field: str,
    allow_missing: bool = False,
) -> None:
    """Raise when a provenance identifier occurs in both partitions."""
    if not allow_missing:
        missing = [row for row in [*train, *test] if row.get(field) is None or not str(row[field]).strip()]
        if missing:
            raise ValueError(f"Split contains rows without required field {field!r}")

    overlap = _identifier_values(train, field) & _identifier_values(test, field)
    if overlap:
        preview = ", ".join(sorted(overlap)[:5])
        raise ValueError(f"Unsafe split: {len(overlap)} {field!r} values occur in both partitions: {preview}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Split JSONL data by source document without train/test leakage.")
    ap.add_argument("--input", required=True, help="Input JSONL dataset")
    ap.add_argument("--train-output", required=True, help="Output path for training set")
    ap.add_argument("--test-output", required=True, help="Output path for test set")
    ap.add_argument("--test-fraction", type=float, default=0.2, help="Fraction for test set (default: 0.2)")
    ap.add_argument("--stratify-field", default="task_name", help="Field used for distribution reporting")
    ap.add_argument(
        "--source-field",
        default="document_id",
        help="Required source-group field (default: document_id)",
    )
    ap.add_argument(
        "--manifest-output",
        default=None,
        help="Optional path for a split provenance manifest",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    rows = load_jsonl(Path(args.input))
    if not rows:
        raise SystemExit("No rows to split.")

    train, test = stratified_split(
        rows,
        test_fraction=args.test_fraction,
        stratify_field=args.stratify_field,
        source_field=args.source_field,
        seed=args.seed,
    )

    write_jsonl(Path(args.train_output), train)
    write_jsonl(Path(args.test_output), test)

    # Summary
    train_dist = Counter(str(r.get(args.stratify_field, "unknown")) for r in train)
    test_dist = Counter(str(r.get(args.stratify_field, "unknown")) for r in test)
    train_sources = _identifier_values(train, args.source_field)
    test_sources = _identifier_values(test, args.source_field)
    summary = {
        "schema_version": 1,
        "method": "source_grouped",
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "source_field": args.source_field,
        "stratify_field": args.stratify_field,
        "total": len(rows),
        "train": len(train),
        "test": len(test),
        "achieved_test_fraction": round(len(test) / max(1, len(rows)), 6),
        "train_sources": len(train_sources),
        "test_sources": len(test_sources),
        "source_overlap": sorted(train_sources & test_sources),
        "train_distribution": dict(train_dist),
        "test_distribution": dict(test_dist),
        "artifacts": {
            "input": {"path": args.input, "sha256": _sha256_file(Path(args.input))},
            "train": {
                "path": args.train_output,
                "sha256": _sha256_file(Path(args.train_output)),
            },
            "test": {
                "path": args.test_output,
                "sha256": _sha256_file(Path(args.test_output)),
            },
        },
    }

    if args.manifest_output:
        manifest_path = Path(args.manifest_output)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
