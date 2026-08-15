"""Typed contracts and content identities for Forge pipeline stages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


def json_value(value: Any) -> JsonValue:
    """Convert supported configuration values into canonical JSON data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return json_value(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_value(item) for item in value]
    raise TypeError(f"Value is not JSON serialisable: {type(value).__name__}")


def canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible data with stable ordering and separators."""
    payload = json.dumps(
        json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PathFingerprint:
    """Content identity for a file or recursively hashed directory."""

    kind: str
    sha256: str
    bytes: int
    files: int


def fingerprint_path(path: Path) -> PathFingerprint:
    """Fingerprint a file or directory without including its absolute location."""
    if not path.exists():
        raise FileNotFoundError(f"Pipeline input does not exist: {path}")
    if path.is_file():
        return PathFingerprint(kind="file", sha256=sha256_file(path), bytes=path.stat().st_size, files=1)
    if not path.is_dir():
        raise ValueError(f"Unsupported pipeline input type: {path}")

    entries: list[dict[str, JsonValue]] = []
    total_bytes = 0
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = candidate.relative_to(path).as_posix()
        size = candidate.stat().st_size
        total_bytes += size
        entries.append({"path": relative, "bytes": size, "sha256": sha256_file(candidate)})
    return PathFingerprint(
        kind="directory",
        sha256=canonical_sha256(entries),
        bytes=total_bytes,
        files=len(entries),
    )


@dataclass(frozen=True)
class ArtifactBinding:
    """A named stage input or output path."""

    role: str
    path: str
    scope: str = "run"
    media_type: str = "application/octet-stream"

    def resolve(self, run_dir: Path) -> Path:
        if self.scope == "run":
            return run_dir / self.path
        if self.scope == "external":
            return Path(self.path).expanduser().resolve()
        raise ValueError(f"Unknown artifact scope: {self.scope!r}")


@dataclass(frozen=True)
class ArtifactRef:
    """Immutable evidence recorded for a concrete pipeline artifact."""

    role: str
    path: str
    scope: str
    media_type: str
    kind: str
    sha256: str
    bytes: int
    files: int

    @classmethod
    def capture(cls, binding: ArtifactBinding, run_dir: Path) -> "ArtifactRef":
        resolved = binding.resolve(run_dir)
        identity = fingerprint_path(resolved)
        return cls(
            role=binding.role,
            path=binding.path,
            scope=binding.scope,
            media_type=binding.media_type,
            kind=identity.kind,
            sha256=identity.sha256,
            bytes=identity.bytes,
            files=identity.files,
        )

    def verify(self, run_dir: Path) -> bool:
        binding = ArtifactBinding(self.role, self.path, self.scope, self.media_type)
        try:
            current = ArtifactRef.capture(binding, run_dir)
        except (FileNotFoundError, OSError, ValueError):
            return False
        return current.sha256 == self.sha256 and current.bytes == self.bytes and current.files == self.files


@dataclass(frozen=True)
class ModelRef:
    provider: str
    name: str
    revision: str = "unversioned"
    parameters: Mapping[str, JsonValue] | None = None


@dataclass(frozen=True)
class PromptRef:
    name: str
    version: str
    sha256: str


@dataclass(frozen=True)
class IngestConfig:
    sources: tuple[str, ...]


@dataclass(frozen=True)
class SourceGovernanceConfig:
    input_path: str = "documents.jsonl"
    output_path: str = "governed_documents.jsonl"
    rejected_path: str = "rejected_documents.jsonl"
    report_path: str = "source_governance_report.json"
    source_manifest: str | None = None
    required_use: str = "training"
    pii_action: str = "reject"
    allow_unknown_rights: bool = True


@dataclass(frozen=True)
class GenerationConfig:
    tasks: tuple[str, ...]
    model: str
    max_examples: int
    max_chars: int
    overlap: int
    dry_run: bool
    input_path: str = "governed_documents.jsonl"


@dataclass(frozen=True)
class QualityConfig:
    input_path: str = "raw_dataset.jsonl"
    output_path: str = "quality.jsonl"


@dataclass(frozen=True)
class RecordGovernanceConfig:
    input_path: str = "quality.jsonl"
    output_path: str = "governed_records.jsonl"
    rejected_path: str = "rejected_records.jsonl"
    report_path: str = "record_governance_report.json"
    pii_action: str = "reject"
    text_fields: tuple[str, ...] = ("input_text", "output_text")


@dataclass(frozen=True)
class DedupeConfig:
    input_path: str = "governed_records.jsonl"
    output_path: str = "deduped.jsonl"
    rejected_path: str = "dedupe_rejections.jsonl"
    report_path: str = "dedupe_report.json"
    text_fields: tuple[str, ...] = ("input_text", "output_text")
    shingle_size: int = 3
    minhash_permutations: int = 64
    lsh_bands: int = 32
    fuzzy_threshold: float = 0.8
    semantic_backend: str = "disabled"
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_revision: str = "main"
    semantic_threshold: float = 0.9
    semantic_max_records: int = 5000


@dataclass(frozen=True)
class JudgmentConfig:
    input_path: str = "deduped.jsonl"
    output_path: str = "judged.jsonl"
    model: str = "gpt-4.1-mini"
    max_concurrent: int = 5
    dry_run: bool = False


@dataclass(frozen=True)
class ContaminationConfig:
    input_path: str = "judged.jsonl"
    benchmark_paths: tuple[str, ...] = ()
    output_path: str = "contamination_report.json"
    text_fields: tuple[str, ...] = ("output_text", "input_text")
    shingle_size: int = 3
    fuzzy_threshold: float = 0.8
    semantic_backend: str = "disabled"
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_revision: str = "main"
    semantic_threshold: float = 0.9
    semantic_max_comparisons: int = 250_000
    fail_on_contamination: bool = True


@dataclass(frozen=True)
class DifficultyConfig:
    input_path: str = "judged.jsonl"
    output_path: str = "difficulty.jsonl"


@dataclass(frozen=True)
class SelectionConfig:
    input_path: str = "difficulty.jsonl"
    output_path: str = "selected.jsonl"
    count: int | None = None
    strategy: str = "quality_weighted"


@dataclass(frozen=True)
class SplitConfig:
    input_path: str = "selected.jsonl"
    train_path: str = "train.jsonl"
    test_path: str = "test.jsonl"
    manifest_path: str = "split_manifest.json"
    test_fraction: float = 0.2
    source_field: str = "document_id"
    stratify_field: str = "task_name"
    seed: int = 42


@dataclass(frozen=True)
class ProfileConfig:
    train_path: str = "train.jsonl"
    test_path: str = "test.jsonl"
    source_governance_path: str = "source_governance_report.json"
    record_governance_path: str = "record_governance_report.json"
    dedupe_report_path: str = "dedupe_report.json"
    contamination_path: str = "contamination_report.json"
    rejected_sources_path: str = "rejected_documents.jsonl"
    rejected_records_path: str = "rejected_records.jsonl"
    dedupe_rejections_path: str = "dedupe_rejections.jsonl"
    output_path: str = "dataset_profile.json"


@dataclass(frozen=True)
class TrainingConfig:
    """Backend-neutral contract for a future training stage adapter."""

    train_path: str
    validation_path: str
    model: str
    backend: str
    output_path: str
    seed: int
    parameters: Mapping[str, JsonValue]


@dataclass(frozen=True)
class EvaluationConfig:
    """Backend-neutral contract for a future evaluation stage adapter."""

    benchmark_path: str
    model_artifact_path: str
    output_path: str
    metrics: tuple[str, ...]
    seed: int
