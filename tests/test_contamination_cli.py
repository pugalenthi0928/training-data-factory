"""CLI tests for the mandatory contamination gate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_benchmark_input_is_required(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_jsonl(dataset, [{"output_text": "A generated example"}])

    result = subprocess.run(
        [sys.executable, "scripts/check_contamination.py", "--input", str(dataset)],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "provide at least one" in result.stderr


def test_missing_benchmark_fails_closed(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_jsonl(dataset, [{"output_text": "A generated example"}])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_contamination.py",
            "--input",
            str(dataset),
            "--benchmark",
            str(tmp_path / "missing.jsonl"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Benchmark file not found" in result.stderr


def test_flagged_overlap_fails_gate_and_writes_report(tmp_path: Path) -> None:
    shared = "The quick brown fox jumps over the lazy dog and then runs away very fast"
    dataset = tmp_path / "dataset.jsonl"
    benchmark = tmp_path / "benchmark.jsonl"
    report = tmp_path / "report.json"
    _write_jsonl(dataset, [{"output_text": shared}])
    _write_jsonl(benchmark, [{"question": shared}])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_contamination.py",
            "--input",
            str(dataset),
            "--benchmark",
            str(benchmark),
            "--output",
            str(report),
            "--fail-on-contamination",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert json.loads(report.read_text(encoding="utf-8"))["contaminated_count"] == 1


def test_independent_text_passes_gate(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    benchmark = tmp_path / "benchmark.jsonl"
    _write_jsonl(dataset, [{"output_text": "A short generated response"}])
    _write_jsonl(
        benchmark,
        [
            {
                "question": "How should a payment service preserve consistency when concurrent transactions update one account?"
            }
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_contamination.py",
            "--input",
            str(dataset),
            "--benchmark",
            str(benchmark),
            "--fail-on-contamination",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
