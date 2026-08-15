"""Frozen evaluation releases, blinded review packets, and human calibration metrics."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .contracts import canonical_sha256, sha256_file

EVALUATION_SCHEMA_VERSION = "forge.evaluation-release/v1"
ANNOTATION_PROTOCOL_VERSION = "forge.annotation-protocol/v1"
CALIBRATION_SCHEMA_VERSION = "forge.human-calibration/v1"
ANNOTATION_REASON_CODES = (
    "unsupported_by_source",
    "incorrect",
    "incomplete",
    "instruction_mismatch",
    "ambiguous",
    "privacy_risk",
    "unsafe_content",
    "verbosity_without_value",
    "clearer_reasoning",
    "better_source_support",
    "both_acceptable",
    "both_unacceptable",
)

_ITEM_FIELDS = (
    "item_id",
    "task",
    "slice",
    "source_id",
    "source_excerpt",
    "prompt",
    "reference_answer",
    "forge_candidate",
    "baseline_candidate",
)


class EvaluationValidationError(ValueError):
    """Raised when evaluation evidence violates its declared contract."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationValidationError(f"evaluation artifact is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationValidationError(f"evaluation artifact is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluationValidationError(f"evaluation artifact must be a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise EvaluationValidationError(f"evaluation artifact is missing: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationValidationError(f"{path} contains invalid JSON on line {line_number}") from exc
        if not isinstance(value, dict):
            raise EvaluationValidationError(f"{path} line {line_number} must be a JSON object")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _artifact(path: Path, root: Path, *, records: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if records is not None:
        value["records"] = records
    return value


def _identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = manifest.get("artifacts", {})
    return {
        "schema_version": manifest.get("schema_version"),
        "protocol_version": manifest.get("protocol_version"),
        "seed": manifest.get("seed"),
        "target_items": manifest.get("target_items"),
        "minimum_overlap_items": manifest.get("minimum_overlap_items"),
        "authoring": manifest.get("authoring"),
        "generator_families": manifest.get("generator_families"),
        "counts": manifest.get("counts"),
        "artifacts": {
            role: value.get("sha256") for role, value in sorted(artifacts.items()) if isinstance(value, Mapping)
        },
    }


def _validate_items(items: Sequence[Mapping[str, Any]]) -> None:
    if not items:
        raise EvaluationValidationError("evaluation set is empty")
    seen: set[str] = set()
    for index, item in enumerate(items, start=1):
        missing = [field for field in _ITEM_FIELDS if not str(item.get(field, "")).strip()]
        if missing:
            raise EvaluationValidationError(f"evaluation item {index} is missing: {', '.join(missing)}")
        item_id = str(item["item_id"])
        if item_id in seen:
            raise EvaluationValidationError(f"duplicate evaluation item_id: {item_id}")
        seen.add(item_id)
        if str(item["forge_candidate"]).strip() == str(item["baseline_candidate"]).strip():
            raise EvaluationValidationError(f"evaluation item {item_id} has identical candidates")


def _primary_positions(item_id: str, seed: int) -> dict[str, str]:
    flip = int(canonical_sha256({"item_id": item_id, "seed": seed})[:8], 16) % 2 == 1
    return {"A": "baseline", "B": "forge"} if flip else {"A": "forge", "B": "baseline"}


def _presentation(item: Mapping[str, Any], positions: Mapping[str, str], variant: str) -> dict[str, Any]:
    candidates = {
        "forge": str(item["forge_candidate"]),
        "baseline": str(item["baseline_candidate"]),
    }
    item_id = str(item["item_id"])
    return {
        "schema_version": "forge.blind-presentation/v1",
        "presentation_id": f"blind_{item_id}_{variant}",
        "item_id": item_id,
        "variant": variant,
        "task": str(item["task"]),
        "slice": str(item["slice"]),
        "source_id": str(item["source_id"]),
        "source_excerpt": str(item["source_excerpt"]),
        "prompt": str(item["prompt"]),
        "reference_answer": str(item["reference_answer"]),
        "candidate_a": candidates[positions["A"]],
        "candidate_b": candidates[positions["B"]],
        "protocol_version": ANNOTATION_PROTOCOL_VERSION,
    }


def _write_review_sheet(path: Path, packet: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "presentation_id",
        "item_id",
        "task",
        "slice",
        "source_excerpt",
        "prompt",
        "reference_answer",
        "candidate_a",
        "candidate_b",
        "preference",
        "confidence",
        "reason_codes",
        "notes",
        "annotator_id",
        "reviewer_type",
        "protocol_version",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in packet:
            writer.writerow(
                {
                    **{field: row.get(field, "") for field in fields},
                    "preference": "",
                    "confidence": "",
                    "reason_codes": "",
                    "notes": "",
                    "annotator_id": "",
                    "reviewer_type": "human",
                }
            )


def freeze_evaluation_set(
    items_path: Path,
    protocol_path: Path,
    output_dir: Path,
    *,
    author: str,
    origin: str,
    independence_status: str,
    generator_families: Sequence[str],
    seed: int = 42,
    target_items: int = 200,
    minimum_overlap_items: int = 50,
) -> dict[str, Any]:
    """Freeze evaluation inputs and build separate human and swapped judge packets."""
    if independence_status not in {"independent", "controlled_fixture"}:
        raise EvaluationValidationError("independence_status must be independent or controlled_fixture")
    if not author.strip() or not origin.strip():
        raise EvaluationValidationError("evaluation author and origin are required")
    if target_items < 1 or minimum_overlap_items < 1:
        raise EvaluationValidationError("evaluation targets must be positive")
    items = _read_jsonl(items_path)
    _validate_items(items)
    protocol = protocol_path.read_text(encoding="utf-8")
    if ANNOTATION_PROTOCOL_VERSION not in protocol:
        raise EvaluationValidationError(f"protocol must declare {ANNOTATION_PROTOCOL_VERSION}")

    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_items = output_dir / "evaluation_items.jsonl"
    frozen_protocol = output_dir / "annotation_protocol.md"
    human_packet_path = output_dir / "human_review_packet.jsonl"
    judge_packet_path = output_dir / "judge_review_packet.jsonl"
    review_sheet_path = output_dir / "human_review_sheet.csv"
    key_path = output_dir / "evaluation_blinding_key.json"

    _write_jsonl(frozen_items, items)
    frozen_protocol.write_text(protocol, encoding="utf-8")

    human_packet: list[dict[str, Any]] = []
    judge_packet: list[dict[str, Any]] = []
    key_entries: list[dict[str, Any]] = []
    for item in items:
        item_id = str(item["item_id"])
        primary = _primary_positions(item_id, seed)
        swapped = {"A": primary["B"], "B": primary["A"]}
        primary_presentation = _presentation(item, primary, "primary")
        swapped_presentation = _presentation(item, swapped, "swapped")
        human_packet.append(primary_presentation)
        judge_packet.extend((primary_presentation, swapped_presentation))
        for variant, positions in (("primary", primary), ("swapped", swapped)):
            key_entries.append(
                {
                    "presentation_id": f"blind_{item_id}_{variant}",
                    "item_id": item_id,
                    "variant": variant,
                    "slice": str(item["slice"]),
                    "positions": positions,
                }
            )

    _write_jsonl(human_packet_path, human_packet)
    _write_jsonl(judge_packet_path, judge_packet)
    _write_review_sheet(review_sheet_path, human_packet)
    key_path.write_text(
        json.dumps(
            {
                "schema_version": "forge.blinding-key/v1",
                "warning": "Keep this file from annotators until blind review is complete.",
                "entries": key_entries,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = {
        "evaluation_items": _artifact(frozen_items, output_dir, records=len(items)),
        "annotation_protocol": _artifact(frozen_protocol, output_dir),
        "human_review_packet": _artifact(human_packet_path, output_dir, records=len(human_packet)),
        "judge_review_packet": _artifact(judge_packet_path, output_dir, records=len(judge_packet)),
        "human_review_sheet": _artifact(review_sheet_path, output_dir, records=len(human_packet)),
        "blinding_key": _artifact(key_path, output_dir, records=len(key_entries)),
    }
    status = (
        "frozen_candidate"
        if independence_status == "independent" and len(items) >= target_items
        else "controlled_fixture"
    )
    manifest: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "protocol_version": ANNOTATION_PROTOCOL_VERSION,
        "created_at": _now(),
        "status": status,
        "seed": seed,
        "target_items": target_items,
        "minimum_overlap_items": minimum_overlap_items,
        "authoring": {
            "author": author,
            "origin": origin,
            "independence_status": independence_status,
            "generator_context_excluded": independence_status == "independent",
        },
        "generator_families": sorted(set(generator_families)),
        "counts": {
            "items": len(items),
            "human_presentations": len(human_packet),
            "judge_presentations": len(judge_packet),
            "slices": dict(sorted(Counter(str(item["slice"]) for item in items).items())),
            "tasks": dict(sorted(Counter(str(item["task"]) for item in items).items())),
        },
        "artifacts": artifacts,
        "claim_status": {
            "independent_evaluation": "frozen" if status == "frozen_candidate" else "controlled_fixture_only",
            "human_calibration": "not_yet_collected",
            "model_quality": "not_established",
        },
    }
    manifest["evaluation_id"] = f"forge_eval_{canonical_sha256(_identity(manifest))[:20]}"
    manifest_path = output_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    verify_evaluation_release(manifest_path)
    return manifest


def verify_evaluation_release(manifest_path: Path) -> dict[str, Any]:
    """Verify every frozen evaluation artifact and the content-addressed ID."""
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise EvaluationValidationError("unsupported evaluation release schema")
    root = manifest_path.parent.resolve()
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise EvaluationValidationError("evaluation release has no artifacts")
    verified = 0
    for role, value in artifacts.items():
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            raise EvaluationValidationError(f"invalid evaluation artifact entry: {role}")
        path = (root / value["path"]).resolve()
        if not path.is_relative_to(root):
            raise EvaluationValidationError(f"evaluation artifact escapes release directory: {role}")
        if not path.is_file() or sha256_file(path) != value.get("sha256"):
            raise EvaluationValidationError(f"evaluation artifact hash mismatch: {role}")
        if path.stat().st_size != value.get("bytes"):
            raise EvaluationValidationError(f"evaluation artifact size mismatch: {role}")
        if isinstance(value.get("records"), int):
            if path.suffix == ".jsonl" and len(_read_jsonl(path)) != value["records"]:
                raise EvaluationValidationError(f"evaluation artifact row count mismatch: {role}")
            if path.suffix == ".csv":
                with path.open("r", encoding="utf-8", newline="") as handle:
                    if sum(1 for _ in csv.DictReader(handle)) != value["records"]:
                        raise EvaluationValidationError(f"evaluation artifact row count mismatch: {role}")
        verified += 1
    expected_id = f"forge_eval_{canonical_sha256(_identity(manifest))[:20]}"
    if manifest.get("evaluation_id") != expected_id:
        raise EvaluationValidationError("evaluation release identity mismatch")
    return {"verified": True, "evaluation_id": expected_id, "artifacts": verified}


def cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float:
    """Return chance-corrected agreement for two complete nominal label sequences."""
    if len(left) != len(right) or not left:
        raise EvaluationValidationError("Cohen's kappa requires equal non-empty label sequences")
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    labels = set(left_counts) | set(right_counts)
    expected = sum((left_counts[label] / len(left)) * (right_counts[label] / len(right)) for label in labels)
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return round((observed - expected) / (1.0 - expected), 6)


def krippendorff_alpha_nominal(units: Sequence[Sequence[str]]) -> float | None:
    """Return nominal Krippendorff alpha for variable raters and missing units."""
    usable = [list(unit) for unit in units if len(unit) >= 2]
    if not usable:
        return None
    marginal: Counter[str] = Counter()
    observed_disagreement = 0.0
    total_values = 0
    for unit in usable:
        counts = Counter(unit)
        size = len(unit)
        marginal.update(counts)
        total_values += size
        disagreements = size * size - sum(count * count for count in counts.values())
        observed_disagreement += disagreements / (size - 1)
    observed = observed_disagreement / total_values
    if total_values < 2:
        return None
    expected = (total_values * total_values - sum(count * count for count in marginal.values())) / (
        total_values * (total_values - 1)
    )
    if math.isclose(expected, 0.0):
        return 1.0 if math.isclose(observed, 0.0) else None
    return round(1.0 - observed / expected, 6)


def _normalise_preference(record: Mapping[str, Any], key: Mapping[str, Any]) -> str:
    preference = str(record.get("preference", "")).strip()
    if preference in {"tie", "both_bad"}:
        return preference
    if preference not in {"A", "B"}:
        raise EvaluationValidationError(f"invalid preference for {record.get('presentation_id')}: {preference}")
    positions = key.get("positions")
    if not isinstance(positions, Mapping) or positions.get(preference) not in {"forge", "baseline"}:
        raise EvaluationValidationError("blinding key has invalid position mapping")
    return str(positions[preference])


def _consensus(labels: Sequence[str]) -> str:
    counts = Counter(labels)
    ordered = counts.most_common()
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return "tie"
    return ordered[0][0]


def _macro_f1(expected: Sequence[str], predicted: Sequence[str]) -> float:
    labels = sorted(set(expected) | set(predicted))
    values = []
    for label in labels:
        true_positive = sum(a == label and b == label for a, b in zip(expected, predicted))
        false_positive = sum(a != label and b == label for a, b in zip(expected, predicted))
        false_negative = sum(a == label and b != label for a, b in zip(expected, predicted))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return round(sum(values) / len(values), 6) if values else 0.0


def _bootstrap_interval(
    values: Sequence[float], *, seeds: Sequence[int] = (17, 42, 97), resamples_per_seed: int = 200
) -> dict[str, Any]:
    if not values:
        return {"method": "paired_nonparametric_bootstrap", "lower": None, "upper": None}
    estimates: list[float] = []
    for seed in seeds:
        random_state = random.Random(seed)
        for _ in range(resamples_per_seed):
            sample = [values[random_state.randrange(len(values))] for _ in values]
            estimates.append(sum(sample) / len(sample))
    estimates.sort()
    lower_index = max(0, int(0.025 * len(estimates)) - 1)
    upper_index = min(len(estimates) - 1, int(0.975 * len(estimates)))
    return {
        "method": "paired_nonparametric_bootstrap",
        "confidence": 0.95,
        "seeds": list(seeds),
        "resamples_per_seed": resamples_per_seed,
        "lower": round(estimates[lower_index], 6),
        "upper": round(estimates[upper_index], 6),
    }


def _pairwise_summary(consensus: Mapping[str, str], item_slices: Mapping[str, str]) -> dict[str, Any]:
    labels = list(consensus.values())
    forge_wins = labels.count("forge")
    baseline_wins = labels.count("baseline")
    ties = labels.count("tie")
    both_bad = labels.count("both_bad")
    scores = [
        1.0 if label == "forge" else 0.0 if label == "baseline" else 0.5 for label in labels if label != "both_bad"
    ]
    slices: dict[str, list[float]] = defaultdict(list)
    for item_id, label in consensus.items():
        if label != "both_bad":
            slices[item_slices[item_id]].append(1.0 if label == "forge" else 0.0 if label == "baseline" else 0.5)
    return {
        "items": len(labels),
        "forge_wins": forge_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "both_bad": both_bad,
        "forge_win_rate_including_ties": round(sum(scores) / len(scores), 6) if scores else None,
        "win_rate_interval": _bootstrap_interval(scores),
        "by_slice": {
            name: {
                "items": len(values),
                "forge_win_rate_including_ties": round(sum(values) / len(values), 6),
            }
            for name, values in sorted(slices.items())
        },
    }


def _load_annotations(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() != ".csv":
        return _read_jsonl(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle) if str(row.get("preference", "")).strip()]


def _annotation_reason_codes(value: Any) -> list[str]:
    if isinstance(value, str):
        codes = [code.strip() for code in value.split("|") if code.strip()]
    elif isinstance(value, list) and all(isinstance(code, str) for code in value):
        codes = value
    else:
        raise EvaluationValidationError("annotation reason_codes must be a list or pipe-delimited string")
    if any(code not in ANNOTATION_REASON_CODES for code in codes):
        raise EvaluationValidationError("annotation contains an unknown reason code")
    return codes


def _annotation_confidence(value: Any) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationValidationError("annotation confidence must be 1, 2, or 3") from exc
    if not numeric.is_integer() or int(numeric) not in {1, 2, 3}:
        raise EvaluationValidationError("annotation confidence must be 1, 2, or 3")
    return int(numeric)


def _expected_calibration_error(correct: Sequence[bool], confidence: Sequence[float], bins: int = 5) -> float | None:
    if not correct or len(correct) != len(confidence):
        return None
    total = len(correct)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            i
            for i, value in enumerate(confidence)
            if (lower <= value <= upper if index == bins - 1 else lower <= value < upper)
        ]
        if not members:
            continue
        accuracy = sum(correct[i] for i in members) / len(members)
        mean_confidence = sum(confidence[i] for i in members) / len(members)
        error += len(members) / total * abs(accuracy - mean_confidence)
    return round(error, 6)


def analyse_evaluation(
    manifest_path: Path,
    annotations_path: Path,
    judge_predictions_path: Path,
    output_path: Path,
    *,
    minimum_alpha: float = 0.667,
    minimum_judge_agreement: float = 0.7,
    minimum_position_consistency: float = 0.8,
) -> dict[str, Any]:
    """Compare blind human preferences with a versioned judge and preserve claim boundaries."""
    verification = verify_evaluation_release(manifest_path)
    manifest = _read_json(manifest_path)
    root = manifest_path.parent
    key_artifact = manifest["artifacts"]["blinding_key"]
    key_data = _read_json(root / key_artifact["path"])
    entries = key_data.get("entries")
    if not isinstance(entries, list):
        raise EvaluationValidationError("blinding key entries are missing")
    key_by_presentation = {
        str(entry["presentation_id"]): entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("presentation_id")
    }
    item_slices = {str(entry["item_id"]): str(entry["slice"]) for entry in entries if isinstance(entry, dict)}

    annotations = _load_annotations(annotations_path)
    if not annotations:
        raise EvaluationValidationError("no completed annotations were supplied")
    units_by_reviewer_type: dict[str, dict[str, list[str]]] = {
        "human": defaultdict(list),
        "fixture": defaultdict(list),
    }
    reviewer_types: Counter[str] = Counter()
    annotators: set[str] = set()
    annotator_types: dict[str, str] = {}
    annotation_keys: set[tuple[str, str]] = set()
    confidence_counts: Counter[int] = Counter()
    reason_code_counts: Counter[str] = Counter()
    for annotation in annotations:
        presentation_id = str(annotation.get("presentation_id", ""))
        key = key_by_presentation.get(presentation_id)
        if key is None or key.get("variant") != "primary":
            raise EvaluationValidationError(f"annotation references an invalid human presentation: {presentation_id}")
        annotator_id = str(annotation.get("annotator_id", "")).strip()
        reviewer_type = str(annotation.get("reviewer_type", "human")).strip()
        if not annotator_id or reviewer_type not in {"human", "fixture"}:
            raise EvaluationValidationError("annotations require a pseudonymous annotator_id and reviewer_type")
        if annotation.get("protocol_version") != ANNOTATION_PROTOCOL_VERSION:
            raise EvaluationValidationError("annotation protocol_version does not match the frozen protocol")
        if annotator_id in annotator_types and annotator_types[annotator_id] != reviewer_type:
            raise EvaluationValidationError("one annotator_id cannot use multiple reviewer types")
        annotation_key = (presentation_id, annotator_id)
        if annotation_key in annotation_keys:
            raise EvaluationValidationError("duplicate annotation from the same reviewer and presentation")
        annotation_keys.add(annotation_key)
        annotator_types[annotator_id] = reviewer_type
        confidence_counts[_annotation_confidence(annotation.get("confidence"))] += 1
        reason_code_counts.update(_annotation_reason_codes(annotation.get("reason_codes", [])))
        label = _normalise_preference(annotation, key)
        item_id = str(key["item_id"])
        units_by_reviewer_type[reviewer_type][item_id].append(label)
        annotators.add(annotator_id)
        reviewer_types[reviewer_type] += 1

    genuine_human_units = units_by_reviewer_type["human"]
    fixture_units = units_by_reviewer_type["fixture"]
    calibration_basis = "human" if genuine_human_units else "fixture"
    analysis_units = genuine_human_units if genuine_human_units else fixture_units
    multiply_reviewed = {item_id: labels for item_id, labels in analysis_units.items() if len(labels) >= 2}
    genuine_human_overlap = {item_id: labels for item_id, labels in genuine_human_units.items() if len(labels) >= 2}
    alpha = krippendorff_alpha_nominal(list(multiply_reviewed.values()))
    genuine_human_alpha = krippendorff_alpha_nominal(list(genuine_human_overlap.values()))
    complete_pairs = [labels for labels in multiply_reviewed.values() if len(labels) == 2]
    kappa = (
        cohen_kappa([labels[0] for labels in complete_pairs], [labels[1] for labels in complete_pairs])
        if complete_pairs
        else None
    )
    consensus = {item_id: _consensus(labels) for item_id, labels in analysis_units.items()}

    target_items = int(manifest["target_items"])
    minimum_overlap_items = int(manifest["minimum_overlap_items"])
    human_reference_ready = (
        manifest.get("status") == "frozen_candidate"
        and len(genuine_human_units) >= target_items
        and len(genuine_human_overlap) >= minimum_overlap_items
        and genuine_human_alpha is not None
        and genuine_human_alpha >= minimum_alpha
    )

    judge_predictions = _read_jsonl(judge_predictions_path)
    judge_by_item: dict[str, dict[str, tuple[str, Mapping[str, Any]]]] = defaultdict(dict)
    judge_models: set[str] = set()
    judge_families: set[str] = set()
    prompt_versions: set[str] = set()
    prompt_hashes: set[str] = set()
    raw_first_choices = 0
    raw_decisions = 0
    for prediction in judge_predictions:
        presentation_id = str(prediction.get("presentation_id", ""))
        key = key_by_presentation.get(presentation_id)
        if key is None:
            raise EvaluationValidationError(f"judge prediction references an unknown presentation: {presentation_id}")
        label = _normalise_preference(prediction, key)
        item_id = str(key["item_id"])
        variant = str(key["variant"])
        if variant in judge_by_item[item_id]:
            raise EvaluationValidationError(f"duplicate judge prediction for {item_id}/{variant}")
        judge_by_item[item_id][variant] = (label, prediction)
        judge_models.add(str(prediction.get("judge_model", "unversioned")))
        judge_families.add(str(prediction.get("judge_family", "unknown")))
        prompt_versions.add(str(prediction.get("prompt_version", "unversioned")))
        prompt_hashes.add(str(prediction.get("prompt_sha256", "unversioned")))
        if prediction.get("preference") in {"A", "B"}:
            raw_decisions += 1
            raw_first_choices += prediction.get("preference") == "A"

    expected: list[str] = []
    predicted: list[str] = []
    correct: list[bool] = []
    confidences: list[float] = []
    position_consistent: list[bool] = []
    for item_id, human_label in consensus.items():
        variants = judge_by_item.get(item_id, {})
        if "primary" in variants:
            judge_label, raw = variants["primary"]
            expected.append(human_label)
            predicted.append(judge_label)
            correct.append(judge_label == human_label)
            raw_confidence = raw.get("confidence")
            try:
                confidence = float(raw_confidence) if isinstance(raw_confidence, (int, float, str)) else -1.0
            except (TypeError, ValueError):
                confidence = -1.0
            if 0.0 <= confidence <= 1.0:
                confidences.append(confidence)
        if "primary" in variants and "swapped" in variants:
            position_consistent.append(variants["primary"][0] == variants["swapped"][0])

    agreement = round(sum(correct) / len(correct), 6) if correct else None
    judge_kappa = cohen_kappa(expected, predicted) if expected else None
    position_consistency = (
        round(sum(position_consistent) / len(position_consistent), 6) if position_consistent else None
    )
    primary_coverage = round(len(expected) / len(consensus), 6) if consensus else 0.0
    reversed_order_coverage = round(len(position_consistent) / len(consensus), 6) if consensus else 0.0
    generator_families = set(str(value) for value in manifest.get("generator_families", []))
    same_family = bool(generator_families & judge_families)
    prompt_hash = next(iter(prompt_hashes), "")
    versioned_judge = (
        len(judge_models) == 1
        and len(judge_families) == 1
        and len(prompt_versions) == 1
        and len(prompt_hashes) == 1
        and "unversioned" not in judge_models
        and "unknown" not in judge_families
        and "unversioned" not in prompt_versions
        and len(prompt_hash) == 64
        and all(character in "0123456789abcdef" for character in prompt_hash.lower())
        and len(confidences) == len(expected)
    )
    headline_eligible = (
        human_reference_ready
        and not same_family
        and versioned_judge
        and primary_coverage == 1.0
        and reversed_order_coverage == 1.0
        and agreement is not None
        and agreement >= minimum_judge_agreement
        and position_consistency is not None
        and position_consistency >= minimum_position_consistency
    )

    report: dict[str, Any] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "created_at": _now(),
        "evaluation_id": verification["evaluation_id"],
        "status": "evaluation_ready" if headline_eligible else "controlled_or_incomplete",
        "input_hashes": {
            "annotations": sha256_file(annotations_path),
            "judge_predictions": sha256_file(judge_predictions_path),
        },
        "human_agreement": {
            "annotations": len(annotations),
            "annotators": len(annotators),
            "reviewer_types": dict(sorted(reviewer_types.items())),
            "confidence_distribution": {str(key): value for key, value in sorted(confidence_counts.items())},
            "reason_codes": dict(sorted(reason_code_counts.items())),
            "calibration_basis": calibration_basis,
            "reviewed_items": len(analysis_units),
            "human_reviewed_items": len(genuine_human_units),
            "multiply_reviewed_items": len(multiply_reviewed),
            "genuine_human_multiply_reviewed_items": len(genuine_human_overlap),
            "krippendorff_alpha_nominal": alpha,
            "genuine_human_krippendorff_alpha_nominal": genuine_human_alpha,
            "cohen_kappa_two_rater_units": kappa,
            "minimum_alpha": minimum_alpha,
            "reference_ready": human_reference_ready,
        },
        "pairwise_comparison": _pairwise_summary(consensus, item_slices),
        "judge_calibration": {
            "judge_models": sorted(judge_models),
            "judge_families": sorted(judge_families),
            "prompt_versions": sorted(prompt_versions),
            "prompt_sha256": sorted(prompt_hashes),
            "human_comparisons": len(expected),
            "exact_agreement": agreement,
            "cohen_kappa": judge_kappa,
            "macro_f1": _macro_f1(expected, predicted),
            "expected_calibration_error": (
                _expected_calibration_error(correct, confidences) if len(confidences) == len(correct) else None
            ),
            "position_consistency": position_consistency,
            "primary_coverage": primary_coverage,
            "reversed_order_coverage": reversed_order_coverage,
            "first_position_choice_rate": round(raw_first_choices / raw_decisions, 6) if raw_decisions else None,
            "same_family_as_generator": same_family,
            "versioned_complete_run": versioned_judge,
            "headline_eligible": headline_eligible,
        },
        "claim_status": {
            "independent_evaluation": manifest["claim_status"]["independent_evaluation"],
            "human_calibration": "established" if human_reference_ready else "not_established",
            "judge_alignment": "measured" if expected else "not_run",
            "model_quality": "not_established_by_this_report",
        },
        "limitations": [
            "Controlled fixture annotations test the analysis path and are not human evidence.",
            "Human agreement must meet the declared coverage and overlap thresholds before evaluation-ready claims.",
            "Judge agreement does not substitute for downstream model evaluation.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
