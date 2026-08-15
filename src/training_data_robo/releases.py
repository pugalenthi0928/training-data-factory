"""Immutable, verifiable dataset release manifests for Forge.

The pipeline produces useful artifacts, but a directory of files is not a
release.  This module turns a completed run into a content-addressed release
with explicit integrity, isolation, and contamination gates.  It also emits a
Croissant JSON-LD description so the resulting dataset is machine-readable.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

RELEASE_SCHEMA_VERSION = "forge.release/v2"
CROISSANT_VERSION = "http://mlcommons.org/croissant/1.1"
_REQUIRED_RUN_FILES = (
    "config.json",
    "pipeline_log.json",
    "source_governance_report.json",
    "record_governance_report.json",
    "dedupe_report.json",
    "contamination_report.json",
    "split_manifest.json",
    "dataset_profile.json",
    "rejected_documents.jsonl",
    "rejected_records.jsonl",
    "dedupe_rejections.jsonl",
    "train.jsonl",
    "test.jsonl",
)


class ReleaseValidationError(ValueError):
    """Raised when a run cannot satisfy the release contract."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl_rows(path: Path) -> int:
    """Count non-empty JSONL records while rejecting malformed rows."""
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReleaseValidationError(f"{path} contains invalid JSON on line {line_number}") from exc
            if not isinstance(value, dict):
                raise ReleaseValidationError(f"{path} line {line_number} must contain a JSON object")
            count += 1
    return count


def _load_json(path: Path, expected_type: type[Any]) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReleaseValidationError(f"required release artifact is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ReleaseValidationError(f"release artifact is not valid JSON: {path}") from exc
    if not isinstance(value, expected_type):
        raise ReleaseValidationError(f"release artifact has the wrong JSON shape: {path}")
    return value


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_input(path_value: str, run_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((run_dir / path, run_dir.parent / path, run_dir.parent.parent / path))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise ReleaseValidationError(f"declared input cannot be found: {path_value}")


def _fingerprint_directory(path: Path) -> Dict[str, Any]:
    files = sorted(
        item for item in path.rglob("*") if item.is_file() and not any(part.startswith(".") for part in item.parts)
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for file_path in files:
        relative = file_path.relative_to(path).as_posix()
        file_digest = sha256_file(file_path)
        size = file_path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\x00")
        total_bytes += size
    return {
        "kind": "directory",
        "name": path.name,
        "sha256": digest.hexdigest(),
        "files": len(files),
        "bytes": total_bytes,
    }


def fingerprint_input(path: Path) -> Dict[str, Any]:
    """Return a portable fingerprint for an input file or directory."""
    if path.is_dir():
        return _fingerprint_directory(path)
    if not path.is_file():
        raise ReleaseValidationError(f"input is neither a file nor a directory: {path}")
    return {
        "kind": "file",
        "name": path.name,
        "sha256": sha256_file(path),
        "files": 1,
        "bytes": path.stat().st_size,
    }


def _collect_inputs(values: Iterable[str], run_dir: Path) -> list[Dict[str, Any]]:
    fingerprints = []
    for value in values:
        resolved = _resolve_input(value, run_dir)
        fingerprint = fingerprint_input(resolved)
        fingerprint["declared_name"] = resolved.name
        fingerprints.append(fingerprint)
    return fingerprints


def _artifact(run_dir: Path, name: str, filename: str) -> Dict[str, Any]:
    path = run_dir / filename
    if not path.is_file():
        raise ReleaseValidationError(f"required release artifact is missing: {path}")
    media_type = "application/x-ndjson" if path.suffix == ".jsonl" else mimetypes.guess_type(path.name)[0]
    result: Dict[str, Any] = {
        "name": name,
        "path": filename,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "media_type": media_type or "application/octet-stream",
    }
    if path.suffix == ".jsonl":
        result["records"] = count_jsonl_rows(path)
    return result


def _validate_pipeline_log(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    failed = [str(entry.get("step", "unknown")) for entry in entries if entry.get("status") != "ok"]
    if failed:
        raise ReleaseValidationError(f"pipeline contains non-passing stages: {', '.join(failed)}")
    if not entries:
        raise ReleaseValidationError("pipeline log is empty")
    return {"name": "pipeline_completion", "status": "passed", "stages": len(entries)}


def _validate_contamination(report: Mapping[str, Any]) -> Dict[str, Any]:
    benchmarks = report.get("benchmarks_checked")
    contaminated = report.get("contaminated_count")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise ReleaseValidationError("contamination report does not identify a benchmark")
    if not isinstance(contaminated, int):
        raise ReleaseValidationError("contamination report has no integer contaminated_count")
    if contaminated != 0:
        raise ReleaseValidationError(f"contamination gate failed with {contaminated} flagged examples")
    return {
        "name": "benchmark_contamination",
        "status": "passed",
        "benchmarks_checked": benchmarks,
        "flagged_examples": contaminated,
        "detectors": report.get("detectors", []),
    }


def _validate_governance(
    source: Mapping[str, Any], record: Mapping[str, Any], dedupe: Mapping[str, Any], release_tier: str
) -> list[Dict[str, Any]]:
    if source.get("status") != "passed":
        raise ReleaseValidationError("source governance report is not passing")
    if record.get("status") != "passed":
        raise ReleaseValidationError("record governance report is not passing")
    if dedupe.get("status") != "passed":
        raise ReleaseValidationError("dedupe report is not passing")
    unknown = source.get("unknown_rights")
    disallowed = source.get("disallowed_rights")
    remaining_pii = record.get("remaining_pii_findings")
    if not isinstance(unknown, int) or not isinstance(disallowed, int):
        raise ReleaseValidationError("source governance report has invalid rights counts")
    if not isinstance(remaining_pii, int):
        raise ReleaseValidationError("record governance report has no remaining PII count")
    if release_tier == "candidate" and unknown:
        raise ReleaseValidationError(f"candidate release has {unknown} sources with unknown usage rights")
    if disallowed:
        raise ReleaseValidationError(f"source governance found {disallowed} sources without permitted training use")
    if remaining_pii:
        raise ReleaseValidationError(f"privacy gate found {remaining_pii} unresolved structured identifiers")
    return [
        {
            "name": "source_rights",
            "status": "passed",
            "unknown_rights": unknown,
            "disallowed_rights": disallowed,
            "candidate_enforced": release_tier == "candidate",
        },
        {
            "name": "record_governance",
            "status": "passed",
            "rejected_records": record.get("rejected_records"),
            "redacted_records": record.get("redacted_records"),
            "remaining_pii_findings": remaining_pii,
        },
        {
            "name": "multi_layer_deduplication",
            "status": "passed",
            "detectors": dedupe.get("detectors", []),
            "dropped_examples": dedupe.get("dropped_examples"),
        },
    ]


def _validate_split(split: Mapping[str, Any], artifacts: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    overlap = split.get("source_overlap")
    if overlap != []:
        raise ReleaseValidationError("source isolation gate failed: train and test sources overlap")
    for partition in ("train", "test"):
        reported_count = split.get(partition)
        actual_count = artifacts[partition].get("records")
        if not isinstance(reported_count, int) or reported_count <= 0:
            raise ReleaseValidationError(f"split manifest has no usable {partition} partition")
        if reported_count != actual_count:
            raise ReleaseValidationError(
                f"{partition} record count mismatch: split manifest={reported_count}, artifact={actual_count}"
            )
        split_artifact = split.get("artifacts", {}).get(partition, {})
        if split_artifact.get("sha256") != artifacts[partition]["sha256"]:
            raise ReleaseValidationError(f"{partition} hash does not match the split manifest")
    return {
        "name": "source_isolation",
        "status": "passed",
        "source_overlap": 0,
        "train_sources": split.get("train_sources"),
        "test_sources": split.get("test_sources"),
    }


def _identity_material(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    artifacts = manifest["artifacts"]
    return {
        "schema_version": manifest["schema_version"],
        "release_tier": manifest["release_tier"],
        "dataset": manifest["dataset"],
        "inputs": {
            group: [
                {
                    "kind": item["kind"],
                    "sha256": item["sha256"],
                    "files": item["files"],
                    "bytes": item["bytes"],
                }
                for item in values
            ]
            for group, values in manifest["inputs"].items()
        },
        "artifacts": {
            name: {
                "sha256": artifact["sha256"],
                "bytes": artifact["bytes"],
                "records": artifact.get("records"),
            }
            for name, artifact in artifacts.items()
            if name
            in {
                "source_governance_report",
                "record_governance_report",
                "dedupe_report",
                "contamination_report",
                "split_manifest",
                "dataset_profile",
                "train",
                "test",
            }
        },
        "gates": manifest["gates"],
    }


def _release_id(manifest: Mapping[str, Any]) -> str:
    return f"forge_{_canonical_hash(_identity_material(manifest))[:20]}"


def build_release_manifest(
    run_dir: Path,
    *,
    dataset_name: str,
    dataset_version: str,
    dataset_license: str,
    benchmark_origin: str,
    release_tier: str = "smoke",
) -> Dict[str, Any]:
    """Validate a completed run and construct its release manifest."""
    run_dir = run_dir.resolve()
    if release_tier not in {"smoke", "candidate"}:
        raise ReleaseValidationError("release_tier must be 'smoke' or 'candidate'")
    required_text = {
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "dataset_license": dataset_license,
        "benchmark_origin": benchmark_origin,
    }
    missing = [name for name, value in required_text.items() if not value.strip()]
    if missing:
        raise ReleaseValidationError(f"release metadata is missing: {', '.join(missing)}")
    if release_tier == "candidate" and dataset_license.strip().lower() in {"unknown", "noassertion", "n/a"}:
        raise ReleaseValidationError("candidate release requires a declared dataset license")
    for filename in _REQUIRED_RUN_FILES:
        if not (run_dir / filename).is_file():
            raise ReleaseValidationError(f"required release artifact is missing: {run_dir / filename}")

    config: Dict[str, Any] = _load_json(run_dir / "config.json", dict)
    runtime_path = run_dir / ".forge" / "runtime_inputs.json"
    runtime_inputs: Dict[str, Any] = _load_json(runtime_path, dict) if runtime_path.is_file() else {}
    pipeline_log: list[Dict[str, Any]] = _load_json(run_dir / "pipeline_log.json", list)
    source_governance: Dict[str, Any] = _load_json(run_dir / "source_governance_report.json", dict)
    record_governance: Dict[str, Any] = _load_json(run_dir / "record_governance_report.json", dict)
    dedupe: Dict[str, Any] = _load_json(run_dir / "dedupe_report.json", dict)
    contamination: Dict[str, Any] = _load_json(run_dir / "contamination_report.json", dict)
    split: Dict[str, Any] = _load_json(run_dir / "split_manifest.json", dict)

    artifacts = {
        "config": _artifact(run_dir, "run configuration", "config.json"),
        "pipeline_log": _artifact(run_dir, "pipeline event log", "pipeline_log.json"),
        "source_governance_report": _artifact(
            run_dir, "source rights and privacy report", "source_governance_report.json"
        ),
        "record_governance_report": _artifact(
            run_dir, "record schema and privacy report", "record_governance_report.json"
        ),
        "dedupe_report": _artifact(run_dir, "multi-layer deduplication report", "dedupe_report.json"),
        "contamination_report": _artifact(run_dir, "contamination report", "contamination_report.json"),
        "split_manifest": _artifact(run_dir, "source-safe split manifest", "split_manifest.json"),
        "dataset_profile": _artifact(run_dir, "dataset profile", "dataset_profile.json"),
        "rejected_documents": _artifact(run_dir, "rejected source documents", "rejected_documents.jsonl"),
        "rejected_records": _artifact(run_dir, "rejected records", "rejected_records.jsonl"),
        "dedupe_rejections": _artifact(run_dir, "duplicate records", "dedupe_rejections.jsonl"),
        "train": _artifact(run_dir, "training split", "train.jsonl"),
        "test": _artifact(run_dir, "test split", "test.jsonl"),
    }
    if (run_dir / "benchmark_results.json").is_file():
        artifacts["benchmark_results"] = _artifact(run_dir, "held-out benchmark results", "benchmark_results.json")

    sources = runtime_inputs.get("sources", config.get("source"))
    benchmarks = runtime_inputs.get("benchmarks", config.get("benchmark_file"))
    if not isinstance(sources, list) or not sources or not all(isinstance(item, str) for item in sources):
        raise ReleaseValidationError("run configuration does not declare source inputs")
    if not isinstance(benchmarks, list) or not benchmarks or not all(isinstance(item, str) for item in benchmarks):
        raise ReleaseValidationError("run configuration does not declare benchmark inputs")

    gates = [
        _validate_pipeline_log(pipeline_log),
        *_validate_governance(source_governance, record_governance, dedupe, release_tier),
        _validate_contamination(contamination),
        _validate_split(split, artifacts),
        {
            "name": "artifact_integrity",
            "status": "passed",
            "hashed_artifacts": len(artifacts),
        },
    ]

    manifest: Dict[str, Any] = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "release_id": "",
        "release_tier": release_tier,
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": dataset_name,
            "version": dataset_version,
            "license": dataset_license,
        },
        "inputs": {
            "sources": _collect_inputs(sources, run_dir),
            "benchmarks": _collect_inputs(benchmarks, run_dir),
            "source_rights": (
                _collect_inputs([runtime_inputs["source_manifest"]], run_dir)
                if isinstance(runtime_inputs.get("source_manifest"), str) and runtime_inputs["source_manifest"]
                else []
            ),
        },
        "benchmark_origin": benchmark_origin,
        "artifacts": artifacts,
        "gates": gates,
        "lineage": [
            {
                "position": index,
                "step": entry.get("step"),
                "status": entry.get("status"),
                "elapsed_seconds": entry.get("elapsed_seconds"),
            }
            for index, entry in enumerate(pipeline_log, start=1)
        ],
        "claim_status": {
            "pipeline_integrity": "established",
            "source_isolation": "established",
            "source_rights": "checked_with_unknowns_allowed" if source_governance["unknown_rights"] else "checked",
            "structured_identifier_privacy": "checked_with_limited_detector_scope",
            "exact_and_fuzzy_deduplication": "checked",
            "semantic_deduplication": "checked" if dedupe.get("semantic_model") else "not_run_in_smoke_release",
            "lexical_contamination": "checked",
            "semantic_contamination": (
                "checked" if contamination.get("semantic_model") else "not_run_in_smoke_release"
            ),
            "human_calibration": "not_yet_completed",
            "model_quality": "not_established_by_this_release",
        },
    }
    manifest["release_id"] = _release_id(manifest)
    return manifest


def build_croissant_metadata(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Create Croissant 1.1 JSON-LD metadata for train and test artifacts."""
    dataset = manifest["dataset"]
    distributions = []
    for partition in ("train", "test"):
        artifact = manifest["artifacts"][partition]
        distributions.append(
            {
                "@type": "cr:FileObject",
                "@id": partition,
                "name": f"{partition} split",
                "contentUrl": artifact["path"],
                "encodingFormat": artifact["media_type"],
                "sha256": artifact["sha256"],
                "contentSize": f"{artifact['bytes']} B",
            }
        )

    return {
        "@context": {
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "prov": "http://www.w3.org/ns/prov#",
            "sc": "https://schema.org/",
        },
        "@type": "sc:Dataset",
        "@id": manifest["release_id"],
        "conformsTo": CROISSANT_VERSION,
        "name": dataset["name"],
        "description": "A source-aware training dataset released through Forge integrity gates.",
        "version": dataset["version"],
        "license": dataset["license"],
        "dateCreated": manifest["created_at"],
        "distribution": distributions,
        "prov:wasGeneratedBy": {
            "@type": "prov:Activity",
            "@id": f"{manifest['release_id']}/pipeline",
            "name": "Forge dataset release pipeline",
            "prov:used": [
                {
                    "@type": "prov:Entity",
                    "@id": f"urn:sha256:{item['sha256']}",
                    "name": item["name"],
                }
                for item in manifest["inputs"]["sources"]
            ],
        },
    }


def verify_release(manifest_path: Path) -> Dict[str, Any]:
    """Recompute release identity and artifact integrity from disk."""
    manifest: Dict[str, Any] = _load_json(manifest_path, dict)
    if manifest.get("schema_version") != RELEASE_SCHEMA_VERSION:
        raise ReleaseValidationError("unsupported release manifest schema")
    expected_id = _release_id(manifest)
    if manifest.get("release_id") != expected_id:
        raise ReleaseValidationError("release identity does not match its content")

    run_dir = manifest_path.resolve().parent
    verified = 0
    for name, artifact in manifest.get("artifacts", {}).items():
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise ReleaseValidationError(f"release artifact entry is malformed: {name}")
        path = run_dir / artifact["path"]
        if not path.is_file():
            raise ReleaseValidationError(f"release artifact is missing: {path}")
        if sha256_file(path) != artifact.get("sha256"):
            raise ReleaseValidationError(f"release artifact hash mismatch: {name}")
        if "records" in artifact and count_jsonl_rows(path) != artifact["records"]:
            raise ReleaseValidationError(f"release artifact record count mismatch: {name}")
        verified += 1
    return {"verified": True, "release_id": expected_id, "artifacts": verified}


def write_release(
    run_dir: Path,
    *,
    dataset_name: str,
    dataset_version: str,
    dataset_license: str,
    benchmark_origin: str,
    release_tier: str = "smoke",
) -> Dict[str, Any]:
    """Build, write, and immediately verify release metadata."""
    manifest = build_release_manifest(
        run_dir,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        dataset_license=dataset_license,
        benchmark_origin=benchmark_origin,
        release_tier=release_tier,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "release_manifest.json"
    croissant_path = run_dir / "croissant.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    croissant_path.write_text(
        json.dumps(build_croissant_metadata(manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    verify_release(manifest_path)
    return manifest
