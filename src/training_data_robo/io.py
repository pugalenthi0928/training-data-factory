"""Consolidated JSONL I/O utilities used across the entire project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

from .logging_config import get_logger

logger = get_logger("training_data_robo.io")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line in %s", path)
                continue
    return records


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Lazily iterate over a JSONL file (for large datasets)."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    """Write a list of dicts as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


def count_jsonl_rows(path: Path) -> int:
    """Count rows without loading entire file into memory."""
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count
