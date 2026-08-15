"""Tests for Forge's content-addressed dataset release contract."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from training_data_robo.releases import (
    CROISSANT_VERSION,
    RELEASE_SCHEMA_VERSION,
    ReleaseValidationError,
    build_release_manifest,
    sha256_file,
    verify_release,
    write_release,
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _make_completed_run(root: Path) -> Path:
    source = root / "source"
    source.mkdir(parents=True)
    (source / "one.txt").write_text("first source", encoding="utf-8")
    (source / "two.txt").write_text("second source", encoding="utf-8")
    benchmark = root / "benchmark.jsonl"
    _write_jsonl(benchmark, [{"question": "independent question"}])

    run = root / "run"
    run.mkdir()
    train_rows = [
        {
            "id": "example_1",
            "document_id": "doc_1",
            "chunk_id": "chunk_1",
            "input_text": "question one",
            "output_text": "answer one",
        },
        {
            "id": "example_2",
            "document_id": "doc_1",
            "chunk_id": "chunk_2",
            "input_text": "question two",
            "output_text": "answer two",
        },
    ]
    test_rows = [
        {
            "id": "example_3",
            "document_id": "doc_2",
            "chunk_id": "chunk_3",
            "input_text": "question three",
            "output_text": "answer three",
        }
    ]
    _write_jsonl(run / "train.jsonl", train_rows)
    _write_jsonl(run / "test.jsonl", test_rows)
    _write_json(
        run / "config.json",
        {
            "source": [str(source)],
            "benchmark_file": [str(benchmark)],
            "dry_run": True,
            "completed_at": "2026-08-15T00:00:00",
        },
    )
    _write_json(
        run / "pipeline_log.json",
        [
            {"step": "Generate training data", "status": "ok", "elapsed_seconds": 1.0},
            {"step": "Contamination check", "status": "ok", "elapsed_seconds": 0.5},
            {"step": "Train/test split", "status": "ok", "elapsed_seconds": 0.2},
        ],
    )
    _write_json(
        run / "contamination_report.json",
        {
            "schema_version": "forge.contamination-report/v2",
            "status": "passed",
            "total_examples": 3,
            "contaminated_count": 0,
            "contamination_rate": 0.0,
            "benchmarks_checked": ["independent"],
            "detectors": ["lexical_8gram", "fuzzy_shingle_jaccard"],
            "semantic_model": None,
        },
    )
    _write_json(
        run / "source_governance_report.json",
        {
            "schema_version": "forge.governance/v1",
            "status": "passed",
            "unknown_rights": 0,
            "disallowed_rights": 0,
            "kept_documents": 2,
            "rejected_documents": 0,
        },
    )
    _write_json(
        run / "record_governance_report.json",
        {
            "schema_version": "forge.governance/v1",
            "status": "passed",
            "rejected_records": 0,
            "redacted_records": 0,
            "remaining_pii_findings": 0,
        },
    )
    _write_json(
        run / "dedupe_report.json",
        {
            "schema_version": "forge.dedupe-report/v1",
            "status": "passed",
            "detectors": ["normalised_exact", "minhash_lsh_jaccard"],
            "dropped_examples": 0,
            "semantic_model": None,
        },
    )
    _write_json(run / "dataset_profile.json", {"schema_version": "forge.dataset-profile/v1", "records": 3})
    _write_jsonl(run / "rejected_documents.jsonl", [])
    _write_jsonl(run / "rejected_records.jsonl", [])
    _write_jsonl(run / "dedupe_rejections.jsonl", [])
    _write_json(
        run / "split_manifest.json",
        {
            "schema_version": 1,
            "method": "source_grouped",
            "train": 2,
            "test": 1,
            "train_sources": 1,
            "test_sources": 1,
            "source_overlap": [],
            "artifacts": {
                "train": {"path": "train.jsonl", "sha256": sha256_file(run / "train.jsonl")},
                "test": {"path": "test.jsonl", "sha256": sha256_file(run / "test.jsonl")},
            },
        },
    )
    return run


def _release(run: Path) -> dict[str, Any]:
    return write_release(
        run,
        dataset_name="forge-test",
        dataset_version="1.0.0",
        dataset_license="MIT",
        benchmark_origin="independently authored test fixture",
        release_tier="candidate",
    )


def test_write_release_creates_verifiable_manifest_and_croissant(tmp_path: Path) -> None:
    run = _make_completed_run(tmp_path)
    manifest = _release(run)

    assert manifest["schema_version"] == RELEASE_SCHEMA_VERSION
    assert manifest["release_id"].startswith("forge_")
    assert manifest["status"] == "passed"
    assert manifest["artifacts"]["train"]["records"] == 2
    assert all(gate["status"] == "passed" for gate in manifest["gates"])
    assert verify_release(run / "release_manifest.json")["verified"] is True

    croissant = json.loads((run / "croissant.json").read_text(encoding="utf-8"))
    assert croissant["conformsTo"] == CROISSANT_VERSION
    assert croissant["@id"] == manifest["release_id"]
    assert {entry["@id"] for entry in croissant["distribution"]} == {"train", "test"}
    assert croissant["prov:wasGeneratedBy"]["name"] == "Forge dataset release pipeline"


def test_release_identity_is_independent_of_run_location_and_timestamp(tmp_path: Path) -> None:
    first = _make_completed_run(tmp_path / "first")
    second_root = tmp_path / "second"
    second_root.mkdir()
    second = second_root / "run"
    shutil.copytree(first, second)
    source = second_root / "source"
    benchmark = second_root / "benchmark.jsonl"
    shutil.copytree(tmp_path / "first" / "source", source)
    shutil.copy2(tmp_path / "first" / "benchmark.jsonl", benchmark)
    config = json.loads((second / "config.json").read_text(encoding="utf-8"))
    config["source"] = [str(source)]
    config["benchmark_file"] = [str(benchmark)]
    config["completed_at"] = "2030-01-01T00:00:00"
    _write_json(second / "config.json", config)

    first_manifest = _release(first)
    second_manifest = _release(second)
    assert first_manifest["release_id"] == second_manifest["release_id"]


def test_contamination_blocks_release(tmp_path: Path) -> None:
    run = _make_completed_run(tmp_path)
    report = json.loads((run / "contamination_report.json").read_text(encoding="utf-8"))
    report["contaminated_count"] = 1
    _write_json(run / "contamination_report.json", report)

    with pytest.raises(ReleaseValidationError, match="contamination gate failed"):
        _release(run)


def test_source_overlap_blocks_release(tmp_path: Path) -> None:
    run = _make_completed_run(tmp_path)
    split = json.loads((run / "split_manifest.json").read_text(encoding="utf-8"))
    split["source_overlap"] = ["doc_1"]
    _write_json(run / "split_manifest.json", split)

    with pytest.raises(ReleaseValidationError, match="source isolation gate failed"):
        _release(run)


def test_candidate_unknown_source_rights_blocks_release(tmp_path: Path) -> None:
    run = _make_completed_run(tmp_path)
    report = json.loads((run / "source_governance_report.json").read_text(encoding="utf-8"))
    report["unknown_rights"] = 1
    _write_json(run / "source_governance_report.json", report)

    with pytest.raises(ReleaseValidationError, match="unknown usage rights"):
        _release(run)


def test_split_hash_mismatch_blocks_release(tmp_path: Path) -> None:
    run = _make_completed_run(tmp_path)
    split = json.loads((run / "split_manifest.json").read_text(encoding="utf-8"))
    split["artifacts"]["train"]["sha256"] = "0" * 64
    _write_json(run / "split_manifest.json", split)

    with pytest.raises(ReleaseValidationError, match="train hash"):
        _release(run)


def test_verify_detects_artifact_tampering(tmp_path: Path) -> None:
    run = _make_completed_run(tmp_path)
    _release(run)
    with (run / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"id": "tampered"}) + "\n")

    with pytest.raises(ReleaseValidationError, match="hash mismatch"):
        verify_release(run / "release_manifest.json")


def test_release_metadata_is_required(tmp_path: Path) -> None:
    run = _make_completed_run(tmp_path)
    with pytest.raises(ReleaseValidationError, match="dataset_license"):
        build_release_manifest(
            run,
            dataset_name="forge-test",
            dataset_version="1.0.0",
            dataset_license="",
            benchmark_origin="independent fixture",
        )


def test_invalid_jsonl_blocks_release(tmp_path: Path) -> None:
    run = _make_completed_run(tmp_path)
    (run / "test.jsonl").write_text("not-json\n", encoding="utf-8")

    with pytest.raises(ReleaseValidationError, match="invalid JSON"):
        _release(run)
