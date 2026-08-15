"""Machine-readable dataset profiles for released Forge artifacts."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import sha256_file


def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
    }


def _audit_reason_counts(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        audit = row.get("forge_audit")
        if not isinstance(audit, dict):
            continue
        decisions = audit.get("decisions")
        if not isinstance(decisions, list):
            continue
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            reasons = decision.get("reason_codes")
            if isinstance(reasons, list):
                counts.update(str(reason) for reason in reasons)
    return counts


def build_dataset_profile(
    *,
    train: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    source_governance: Mapping[str, Any],
    record_governance: Mapping[str, Any],
    dedupe: Mapping[str, Any],
    contamination: Mapping[str, Any],
    rejected_sources: Sequence[Mapping[str, Any]],
    rejected_records: Sequence[Mapping[str, Any]],
    dedupe_rejections: Sequence[Mapping[str, Any]],
    artifact_paths: Mapping[str, Path],
) -> dict[str, Any]:
    records = [*train, *test]
    lengths = [float(len(str(row.get("input_text", ""))) + len(str(row.get("output_text", "")))) for row in records]
    quality = [float(row["quality_score"]) for row in records if isinstance(row.get("quality_score"), (int, float))]
    judge = [float(row["judge_avg_score"]) for row in records if isinstance(row.get("judge_avg_score"), (int, float))]
    rejected = [*rejected_sources, *rejected_records, *dedupe_rejections]
    return {
        "schema_version": "forge.dataset-profile/v1",
        "records": {
            "total": len(records),
            "train": len(train),
            "test": len(test),
            "sources": len({str(row.get("document_id")) for row in records}),
            "tasks": dict(sorted(Counter(str(row.get("task_name", "unknown")) for row in records).items())),
            "difficulty": dict(sorted(Counter(str(row.get("difficulty", "unknown")) for row in records).items())),
            "character_length": _summary(lengths),
            "quality_score": _summary(quality),
            "judge_score": _summary(judge),
        },
        "curation": {
            "source_governance": dict(source_governance),
            "record_governance": dict(record_governance),
            "dedupe": dict(dedupe),
            "contamination": {
                "status": contamination.get("status", "unknown"),
                "flagged_examples": contamination.get("contaminated_count"),
                "detectors": contamination.get("detectors"),
            },
            "rejections": {
                "total": len(rejected),
                "source": len(rejected_sources),
                "record": len(rejected_records),
                "duplicate": len(dedupe_rejections),
                "reason_codes": dict(sorted(_audit_reason_counts(rejected).items())),
            },
        },
        "artifacts": {
            name: {"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in sorted(artifact_paths.items())
        },
        "limitations": [
            "Structured-identifier scanning does not guarantee detection of every personal identifier.",
            "Semantic controls depend on the configured embedding model and calibrated threshold.",
            "Automatic quality and judge scores require human calibration before model-quality claims.",
        ],
    }


def write_profile(path: Path, profile: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
