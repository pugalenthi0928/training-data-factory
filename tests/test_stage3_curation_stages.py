"""Integration tests for Stage 3 dedupe and contamination stage outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import pytest

from forge.contracts import ContaminationConfig, DedupeConfig
from forge.pipeline import StageContext
from forge.similarity import normalise_text
from forge.stages import run_contamination, run_dedupe


class FixtureEncoder:
    name = "fixture-semantic-encoder"
    revision = "1"

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            value = normalise_text(text)
            if "one partition" in value or "same side" in value:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _context(tmp_path: Path, stage: str) -> StageContext:
    return StageContext(tmp_path, "run", stage, "cache")


def _row(row_id: str, output: str) -> dict[str, object]:
    return {
        "id": row_id,
        "document_id": f"doc_{row_id}",
        "chunk_id": f"chunk_{row_id}",
        "task_name": "qa",
        "input_text": "Question",
        "output_text": output,
        "quality_score": 1.0,
    }


def test_semantic_dedupe_quarantines_record_with_reason_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_jsonl(
        tmp_path / "governed_records.jsonl",
        [
            _row("a", "Every source remains in one partition."),
            _row("b", "Keep each document on the same side of the boundary."),
            _row("c", "Weather stations record rainfall."),
        ],
    )
    monkeypatch.setattr("forge.stages.build_encoder", lambda *args: FixtureEncoder())

    report = run_dedupe(
        _context(tmp_path, "dedupe"),
        DedupeConfig(semantic_backend="sentence_transformers", semantic_threshold=0.9),
    )

    assert report["dropped_examples"] == 1
    rejected = json.loads((tmp_path / "dedupe_rejections.jsonl").read_text().splitlines()[0])
    reasons = rejected["forge_audit"]["decisions"][-1]["reason_codes"]
    assert "duplicate.semantic_embedding" in reasons


def test_semantic_contamination_blocks_paraphrase_and_writes_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_jsonl(
        tmp_path / "judged.jsonl",
        [
            _row("a", "Keep each document on the same side of the boundary."),
            _row("b", "Weather stations record rainfall."),
        ],
    )
    benchmark = tmp_path / "benchmark.jsonl"
    _write_jsonl(benchmark, [{"question": "Every source remains in one partition."}])
    monkeypatch.setattr("forge.stages.build_encoder", lambda *args: FixtureEncoder())

    with pytest.raises(ValueError, match="Contamination gate failed"):
        run_contamination(
            _context(tmp_path, "contamination"),
            ContaminationConfig(
                benchmark_paths=(str(benchmark),),
                semantic_backend="sentence_transformers",
                semantic_threshold=0.9,
            ),
        )

    report = json.loads((tmp_path / "contamination_report.json").read_text())
    flagged = next(item for item in report["per_example"] if item["example_id"] == "a")
    assert "contamination.semantic_embedding" in flagged["reason_codes"]
    assert report["status"] == "failed"
