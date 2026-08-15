"""Stage 4 frozen evaluation, blinding, agreement, and judge calibration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.evaluation import (
    EvaluationValidationError,
    analyse_evaluation,
    cohen_kappa,
    freeze_evaluation_set,
    krippendorff_alpha_nominal,
    verify_evaluation_release,
)
from forge.pairwise_judge import (
    PAIRWISE_JUDGE_PROMPT_SHA256,
    PAIRWISE_JUDGE_PROMPT_VERSION,
    build_pairwise_prompt,
    parse_pairwise_judgment,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _items(count: int = 2) -> list[dict[str, object]]:
    return [
        {
            "item_id": f"item_{index}",
            "task": "qa",
            "slice": "support" if index % 2 else "correctness",
            "source_id": f"source_{index}",
            "source_excerpt": f"Verified statement {index}.",
            "prompt": f"What is statement {index}?",
            "reference_answer": f"Verified statement {index}.",
            "forge_candidate": f"Verified statement {index}.",
            "baseline_candidate": f"Unsupported claim {index}.",
        }
        for index in range(1, count + 1)
    ]


def _protocol(path: Path) -> None:
    path.write_text("# Protocol\n\n`forge.annotation-protocol/v1`\n", encoding="utf-8")


def _freeze(tmp_path: Path, *, independent: bool = False) -> tuple[Path, dict[str, object]]:
    items = tmp_path / "items.jsonl"
    protocol = tmp_path / "protocol.md"
    _write_jsonl(items, _items())
    _protocol(protocol)
    release = tmp_path / "release"
    manifest = freeze_evaluation_set(
        items,
        protocol,
        release,
        author="independent reviewer" if independent else "fixture",
        origin="controlled test",
        independence_status="independent" if independent else "controlled_fixture",
        generator_families=("generator-family",),
        target_items=2,
        minimum_overlap_items=2,
    )
    return release, manifest


def test_nominal_agreement_metrics_are_chance_corrected() -> None:
    assert cohen_kappa(["forge", "baseline", "tie"], ["forge", "baseline", "tie"]) == 1.0
    assert krippendorff_alpha_nominal([["forge", "forge"], ["baseline", "baseline"]]) == 1.0
    assert krippendorff_alpha_nominal([["forge"], ["baseline"]]) is None


def test_pairwise_judge_prompt_and_parser_are_strict() -> None:
    presentation = {
        "presentation_id": "blind_item_primary",
        "task": "qa",
        "source_excerpt": "The verified fact.",
        "prompt": "What is the fact?",
        "reference_answer": "The verified fact.",
        "candidate_a": "The verified fact.",
        "candidate_b": "An unsupported statement.",
    }
    prompt = build_pairwise_prompt(presentation)
    assert "Candidate A" in prompt and "Candidate B" in prompt
    assert PAIRWISE_JUDGE_PROMPT_VERSION == "forge.pairwise-judge/v1"
    assert len(PAIRWISE_JUDGE_PROMPT_SHA256) == 64
    assert (
        parse_pairwise_judgment(
            '{"preference":"A","confidence":0.9,"explanation":"supported","reason_codes":["better_source_support"]}'
        )["preference"]
        == "A"
    )
    with pytest.raises(EvaluationValidationError, match="invalid JSON"):
        parse_pairwise_judgment("Candidate A is better")
    with pytest.raises(EvaluationValidationError, match="invalid confidence"):
        parse_pairwise_judgment('{"preference":"A","confidence":2}')


def test_frozen_release_is_location_independent_and_packets_are_blinded(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first, first_manifest = _freeze(first_root)
    second, second_manifest = _freeze(second_root)

    assert first_manifest["evaluation_id"] == second_manifest["evaluation_id"]
    assert verify_evaluation_release(first / "evaluation_manifest.json")["verified"] is True
    human_rows = [json.loads(line) for line in (first / "human_review_packet.jsonl").read_text().splitlines()]
    judge_rows = [json.loads(line) for line in (first / "judge_review_packet.jsonl").read_text().splitlines()]
    assert len(human_rows) == 2
    assert len(judge_rows) == 4
    assert all(
        "positions" not in row and "forge_candidate" not in row and "baseline_candidate" not in row
        for row in human_rows
    )
    primary = next(row for row in judge_rows if row["item_id"] == "item_1" and row["variant"] == "primary")
    swapped = next(row for row in judge_rows if row["item_id"] == "item_1" and row["variant"] == "swapped")
    assert primary["candidate_a"] == swapped["candidate_b"]
    assert primary["candidate_b"] == swapped["candidate_a"]


def test_tampered_evaluation_artifact_fails_verification(tmp_path: Path) -> None:
    release, _ = _freeze(tmp_path)
    (release / "human_review_packet.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(EvaluationValidationError, match="hash mismatch"):
        verify_evaluation_release(release / "evaluation_manifest.json")


def test_real_human_coverage_and_independent_judge_can_reach_evaluation_ready(tmp_path: Path) -> None:
    release, _ = _freeze(tmp_path, independent=True)
    key = json.loads((release / "evaluation_blinding_key.json").read_text())
    primary = [entry for entry in key["entries"] if entry["variant"] == "primary"]
    annotations: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for entry in key["entries"]:
        preferred = "A" if entry["positions"]["A"] == "forge" else "B"
        predictions.append(
            {
                "presentation_id": entry["presentation_id"],
                "preference": preferred,
                "confidence": 0.95,
                "judge_model": "independent-judge-v1",
                "judge_family": "other-family",
                "prompt_version": PAIRWISE_JUDGE_PROMPT_VERSION,
                "prompt_sha256": PAIRWISE_JUDGE_PROMPT_SHA256,
            }
        )
    for reviewer in ("reviewer_01", "reviewer_02"):
        for entry in primary:
            preferred = "A" if entry["positions"]["A"] == "forge" else "B"
            annotations.append(
                {
                    "presentation_id": entry["presentation_id"],
                    "annotator_id": reviewer,
                    "reviewer_type": "human",
                    "preference": preferred,
                    "confidence": 3,
                    "reason_codes": ["better_source_support"],
                    "protocol_version": "forge.annotation-protocol/v1",
                }
            )
    annotations_path = tmp_path / "annotations.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    report_path = tmp_path / "report.json"
    _write_jsonl(annotations_path, annotations)
    _write_jsonl(predictions_path, predictions)

    report = analyse_evaluation(
        release / "evaluation_manifest.json",
        annotations_path,
        predictions_path,
        report_path,
    )

    assert report["status"] == "evaluation_ready"
    assert report["human_agreement"]["reference_ready"] is True
    assert report["judge_calibration"]["headline_eligible"] is True
    assert report["judge_calibration"]["position_consistency"] == 1.0
    assert "reviewer_01" not in report_path.read_text()


def test_fixture_annotations_remain_outside_human_claims(tmp_path: Path) -> None:
    release, _ = _freeze(tmp_path)
    key = json.loads((release / "evaluation_blinding_key.json").read_text())
    annotations = []
    predictions = []
    for entry in key["entries"]:
        preferred = "A" if entry["positions"]["A"] == "forge" else "B"
        predictions.append(
            {
                "presentation_id": entry["presentation_id"],
                "preference": preferred,
                "confidence": 0.9,
                "judge_model": "same-family-judge",
                "judge_family": "generator-family",
                "prompt_version": PAIRWISE_JUDGE_PROMPT_VERSION,
                "prompt_sha256": PAIRWISE_JUDGE_PROMPT_SHA256,
            }
        )
        if entry["variant"] == "primary":
            for reviewer in ("fixture_01", "fixture_02"):
                annotations.append(
                    {
                        "presentation_id": entry["presentation_id"],
                        "annotator_id": reviewer,
                        "reviewer_type": "fixture",
                        "preference": preferred,
                        "confidence": 3,
                        "reason_codes": ["better_source_support"],
                        "protocol_version": "forge.annotation-protocol/v1",
                    }
                )
    annotations_path = tmp_path / "fixture_annotations.jsonl"
    predictions_path = tmp_path / "fixture_predictions.jsonl"
    _write_jsonl(annotations_path, annotations)
    _write_jsonl(predictions_path, predictions)

    report = analyse_evaluation(
        release / "evaluation_manifest.json",
        annotations_path,
        predictions_path,
        tmp_path / "fixture_report.json",
    )

    assert report["status"] == "controlled_or_incomplete"
    assert report["human_agreement"]["human_reviewed_items"] == 0
    assert report["claim_status"]["human_calibration"] == "not_established"
    assert report["judge_calibration"]["same_family_as_generator"] is True
    assert report["judge_calibration"]["headline_eligible"] is False


def test_fixture_labels_cannot_inflate_genuine_human_overlap(tmp_path: Path) -> None:
    release, _ = _freeze(tmp_path, independent=True)
    key = json.loads((release / "evaluation_blinding_key.json").read_text())
    primary = [entry for entry in key["entries"] if entry["variant"] == "primary"]
    annotations: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    for entry in key["entries"]:
        preferred = "A" if entry["positions"]["A"] == "forge" else "B"
        predictions.append(
            {
                "presentation_id": entry["presentation_id"],
                "preference": preferred,
                "confidence": 0.9,
                "judge_model": "independent-judge-v1",
                "judge_family": "other-family",
                "prompt_version": PAIRWISE_JUDGE_PROMPT_VERSION,
                "prompt_sha256": PAIRWISE_JUDGE_PROMPT_SHA256,
            }
        )
    for entry in primary:
        preferred = "A" if entry["positions"]["A"] == "forge" else "B"
        annotations.extend(
            [
                {
                    "presentation_id": entry["presentation_id"],
                    "annotator_id": "human_01",
                    "reviewer_type": "human",
                    "preference": preferred,
                    "confidence": 3,
                    "reason_codes": ["better_source_support"],
                    "protocol_version": "forge.annotation-protocol/v1",
                },
                {
                    "presentation_id": entry["presentation_id"],
                    "annotator_id": "fixture_01",
                    "reviewer_type": "fixture",
                    "preference": preferred,
                    "confidence": 3,
                    "reason_codes": ["better_source_support"],
                    "protocol_version": "forge.annotation-protocol/v1",
                },
            ]
        )
    annotations_path = tmp_path / "mixed_annotations.jsonl"
    predictions_path = tmp_path / "mixed_predictions.jsonl"
    _write_jsonl(annotations_path, annotations)
    _write_jsonl(predictions_path, predictions)

    report = analyse_evaluation(
        release / "evaluation_manifest.json",
        annotations_path,
        predictions_path,
        tmp_path / "mixed_report.json",
    )

    assert report["human_agreement"]["calibration_basis"] == "human"
    assert report["human_agreement"]["genuine_human_multiply_reviewed_items"] == 0
    assert report["human_agreement"]["reference_ready"] is False
    assert report["judge_calibration"]["headline_eligible"] is False
