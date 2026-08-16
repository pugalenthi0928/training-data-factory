"""Bounded job service for the public Forge demonstration.

The hosted adapter deliberately exposes a small, deterministic surface. It
accepts two text documents, creates controlled local inputs, and invokes the
same ``run_forge`` API used by the CLI and tests. It never accepts paths, URLs,
uploads, model keys, or arbitrary pipeline configuration.
"""

from __future__ import annotations

import hashlib
import io
import json
import secrets
import shutil
import threading
import time
import zipfile
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from training_data_robo.releases import verify_release

from .pipeline import StageExecutionError
from .workflow import ForgeConfig, run_forge

STAGE_ORDER = (
    "ingest",
    "source_governance",
    "generate",
    "quality",
    "record_governance",
    "dedupe",
    "judge",
    "contamination",
    "difficulty",
    "select",
    "split",
    "profile",
)

PUBLIC_ARTIFACTS: Mapping[str, tuple[str, str, str]] = {
    "release": ("release_manifest.json", "Release manifest", "application/json"),
    "profile": ("dataset_profile.json", "Dataset profile", "application/json"),
    "events": ("pipeline_events.jsonl", "Pipeline events", "application/x-ndjson"),
    "pipeline": ("pipeline_log.json", "Pipeline log", "application/json"),
    "source-governance": (
        "source_governance_report.json",
        "Source governance",
        "application/json",
    ),
    "record-governance": (
        "record_governance_report.json",
        "Record governance",
        "application/json",
    ),
    "dedupe": ("dedupe_report.json", "Deduplication report", "application/json"),
    "contamination": (
        "contamination_report.json",
        "Contamination report",
        "application/json",
    ),
    "split": ("split_manifest.json", "Split manifest", "application/json"),
    "train": ("train.jsonl", "Training split", "application/x-ndjson"),
    "test": ("test.jsonl", "Test split", "application/x-ndjson"),
    "rejected-sources": (
        "rejected_documents.jsonl",
        "Rejected sources",
        "application/x-ndjson",
    ),
    "rejected-records": (
        "rejected_records.jsonl",
        "Rejected records",
        "application/x-ndjson",
    ),
    "duplicates": (
        "dedupe_rejections.jsonl",
        "Duplicate quarantine",
        "application/x-ndjson",
    ),
}


class SourceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=12_000)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset: str | None = "release-controls"
    documents: list[SourceDocument] | None = None


@dataclass(frozen=True)
class Preset:
    slug: str
    label: str
    eyebrow: str
    description: str
    documents: tuple[tuple[str, str], tuple[str, str]]

    def public(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "label": self.label,
            "eyebrow": self.eyebrow,
            "description": self.description,
            "documents": [{"title": title, "characters": len(text)} for title, text in self.documents],
        }


PRESETS: Mapping[str, Preset] = {
    "release-controls": Preset(
        slug="release-controls",
        label="AI release controls",
        eyebrow="POLICIES / EVALUATION",
        description="Run two policy documents through source governance, curation, source-isolated splitting and release verification.",
        documents=(
            (
                "Release policy",
                """A production AI release begins with a named owner, a versioned change request and a record of every source used to create the candidate dataset. Each source is checked for permitted use before transformation. Documents that lack a clear decision remain visible in the governance report, and documents that violate policy are quarantined. Generated records retain document and chunk identifiers so an engineer can trace an output back to its source context. The release process records prompt and model references, deterministic configuration, stage inputs, stage outputs and content hashes. A failed stage stops promotion. Passing the pipeline means the declared controls ran successfully on this bounded dataset. It does not by itself establish that a model is accurate, safe or useful in production. Those claims require independent evaluation and human evidence.""",
            ),
            (
                "Evaluation protocol",
                """Evaluation data is prepared independently from training inputs and frozen before candidate results are examined. Review items use stable identifiers and a blinding key so presentation order cannot reveal the candidate system. Human reviewers follow a written annotation protocol, record disagreements and separate calibration items from the final held-out set. Automated judges may assist triage, but their agreement with human labels is measured before their scores are used as evidence. Train and test records are grouped by source to prevent chunks from one document appearing in both partitions. Contamination checks compare generated text with benchmark material before release. The final evidence bundle includes dataset profiles, rejection counts, split hashes, gate outcomes and the limits of the claim. Reproducibility comes from the recorded procedure, not from a polished screenshot.""",
            ),
        ),
    ),
    "incident-operations": Preset(
        slug="incident-operations",
        label="Incident operations",
        eyebrow="RUNBOOKS / PROVENANCE",
        description="Process two incident runbooks while preserving document provenance and source-isolated splitting.",
        documents=(
            (
                "Detection and triage",
                """An incident begins when a monitored service crosses a documented threshold or a customer report is confirmed. The responder records the affected service, observed symptoms, first detection time and current user impact. Triage distinguishes an isolated request failure from a regional or system-wide event. The initial owner opens a durable incident record and links dashboards, deployment changes and relevant logs. If customer data may be exposed, the privacy lead is contacted immediately and access to raw evidence is restricted. Diagnostic actions are time stamped. Responders avoid editing the original evidence and instead attach derived notes with clear authorship. The goal of the first phase is to stabilize the service, preserve useful evidence and communicate what is known without guessing at an unverified root cause.""",
            ),
            (
                "Recovery and review",
                """Recovery is declared only after the primary health indicators return to their agreed range and a separate check confirms that customer workflows succeed. Temporary mitigations are documented with an expiry owner. The incident commander records the decision to close active response and schedules a review while evidence is still available. The review separates contributing conditions from the triggering event and identifies which controls detected, contained or missed the failure. Actions are written as testable changes with one owner and a due date. A follow-up verifies that the change was implemented and that the relevant alert, test or runbook behaves as intended. The review is not used to assign blame. Its purpose is to improve the technical system, the operating procedure and the quality of future decisions.""",
            ),
        ),
    ),
    "regulated-change": Preset(
        slug="regulated-change",
        label="Regulated change",
        eyebrow="CHANGE CONTROL / EVIDENCE",
        description="Run two change-control documents through rights, privacy, quality and release checks.",
        documents=(
            (
                "Change assessment",
                """A regulated system change starts with a description of the intended outcome, the affected process and the evidence needed for approval. The assessor identifies data classes, external dependencies, user groups and any policy obligations that could change the risk. Requirements are linked to test cases before implementation begins. Access follows least privilege and production data is not copied into development fixtures unless an approved control permits it. The change record names the technical owner, the reviewer and the person accountable for release. Known limitations are recorded as decisions rather than hidden in informal messages. If the evidence is incomplete, the change remains in review. Approval confirms that the stated checks were completed for the submitted version, not that every future use of the system is automatically acceptable.""",
            ),
            (
                "Validation and rollback",
                """Validation uses a versioned plan with expected results, objective acceptance criteria and a record of the environment in which each check ran. Test data is separated from production inputs and its origin is documented. Failures are retained with the same care as passing results because they explain how the final decision was reached. Before deployment, the team confirms monitoring, support ownership and a rollback procedure that can be executed within the required recovery window. The release identifier links the approved change, test evidence, deployed artifact and operator instructions. After deployment, a bounded observation period checks the most important service and risk indicators. If a stop condition is reached, the rollback owner acts without waiting for an informal consensus and records the outcome for the post-change review.""",
            ),
        ),
    ),
}


@dataclass(frozen=True)
class HostedSettings:
    data_dir: Path = Path("/tmp/forge-hosted")
    max_workers: int = 1
    max_jobs: int = 24
    ttl_seconds: int = 3_600
    rate_limit: int = 12
    rate_window_seconds: int = 3_600
    min_document_chars: int = 320
    max_document_chars: int = 12_000
    max_total_chars: int = 24_000


@dataclass
class JobRecord:
    job_id: str
    fingerprint: str
    label: str
    run_dir: Path
    documents: tuple[tuple[str, str], tuple[str, str]]
    controlled_sources: bool
    created_epoch: float = field(default_factory=time.time)
    started_epoch: float | None = None
    completed_epoch: float | None = None
    status: str = "queued"
    public_error: str | None = None
    release_id: str | None = None
    verified_artifacts: int | None = None


class HostedValidationError(ValueError):
    """A safe validation error that can be returned to an API caller."""


class HostedRateLimitError(RuntimeError):
    """Raised when a client exceeds the bounded public run allowance."""


def _iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _normalise_text(value: str) -> str:
    return " ".join(value.strip().split())


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _count_jsonl(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


class JobManager:
    """Runs a small number of isolated Forge smoke jobs and exposes safe evidence."""

    def __init__(self, settings: HostedSettings) -> None:
        self.settings = HostedSettings(**{**settings.__dict__, "data_dir": settings.data_dir.expanduser().resolve()})
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(
            max_workers=self.settings.max_workers,
            thread_name_prefix="forge-hosted",
        )
        self._jobs: dict[str, JobRecord] = {}
        self._fingerprints: dict[str, str] = {}
        self._futures: dict[str, Future[None]] = {}
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.RLock()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def presets(self) -> list[dict[str, Any]]:
        return [preset.public() for preset in PRESETS.values()]

    def _resolve_request(self, request: RunRequest) -> tuple[str, tuple[tuple[str, str], tuple[str, str]], bool]:
        if request.documents is not None:
            if request.preset not in {None, "custom"}:
                raise HostedValidationError("Choose a preset or provide custom documents, not both.")
            if len(request.documents) != 2:
                raise HostedValidationError("Custom runs require exactly two source documents.")
            values = tuple((document.title.strip(), _normalise_text(document.text)) for document in request.documents)
            documents = (values[0], values[1])
            label = "Custom source pair"
            controlled_sources = False
        else:
            slug = request.preset or "release-controls"
            preset = PRESETS.get(slug)
            if preset is None:
                raise HostedValidationError("Unknown demonstration preset.")
            label = preset.label
            documents = preset.documents
            controlled_sources = True

        lengths = [len(text) for _, text in documents]
        if any(length < self.settings.min_document_chars for length in lengths):
            raise HostedValidationError(
                f"Each source document needs at least {self.settings.min_document_chars} characters."
            )
        if any(length > self.settings.max_document_chars for length in lengths):
            raise HostedValidationError(
                f"Each source document is limited to {self.settings.max_document_chars} characters."
            )
        if sum(lengths) > self.settings.max_total_chars:
            raise HostedValidationError(
                f"The two source documents are limited to {self.settings.max_total_chars} characters in total."
            )
        if _normalise_text(documents[0][1]).casefold() == _normalise_text(documents[1][1]).casefold():
            raise HostedValidationError("The two source documents must contain different text.")
        return label, documents, controlled_sources

    def _fingerprint(
        self,
        documents: Sequence[tuple[str, str]],
        controlled_sources: bool,
    ) -> str:
        payload = json.dumps(
            {
                "schema": "forge.hosted-request/v1",
                "documents": [{"title": title, "text": _normalise_text(text)} for title, text in documents],
                "controlled_sources": controlled_sources,
                "pipeline": "forge.workflow/v1",
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _check_rate_limit(self, client_key: str, now: float) -> None:
        requests = self._requests[client_key]
        boundary = now - self.settings.rate_window_seconds
        while requests and requests[0] < boundary:
            requests.popleft()
        if len(requests) >= self.settings.rate_limit:
            raise HostedRateLimitError("Run limit reached. Please return later or use the local project.")
        requests.append(now)

    def _remove_job(self, job_id: str) -> None:
        record = self._jobs.pop(job_id, None)
        self._futures.pop(job_id, None)
        if record is None:
            return
        self._fingerprints.pop(record.fingerprint, None)
        if record.run_dir.parent == self.settings.data_dir and record.run_dir.is_dir():
            shutil.rmtree(record.run_dir, ignore_errors=True)

    def _cleanup(self, now: float) -> None:
        expired = [
            record.job_id
            for record in self._jobs.values()
            if record.status in {"succeeded", "failed"}
            and record.completed_epoch is not None
            and now - record.completed_epoch > self.settings.ttl_seconds
        ]
        for job_id in expired:
            self._remove_job(job_id)
        terminal = sorted(
            (record for record in self._jobs.values() if record.status in {"succeeded", "failed"}),
            key=lambda record: record.created_epoch,
        )
        while len(self._jobs) >= self.settings.max_jobs and terminal:
            self._remove_job(terminal.pop(0).job_id)

    def submit(self, request: RunRequest, *, client_key: str) -> tuple[JobRecord, bool]:
        label, documents, controlled_sources = self._resolve_request(request)
        fingerprint = self._fingerprint(documents, controlled_sources)
        now = time.time()
        with self._lock:
            self._cleanup(now)
            existing_id = self._fingerprints.get(fingerprint)
            if existing_id is not None and existing_id in self._jobs:
                return self._jobs[existing_id], True
            if len(self._jobs) >= self.settings.max_jobs:
                raise HostedRateLimitError("The demonstration is at capacity. Please try again shortly.")
            self._check_rate_limit(client_key, now)
            job_id = secrets.token_hex(8)
            run_dir = self.settings.data_dir / job_id
            record = JobRecord(
                job_id=job_id,
                fingerprint=fingerprint,
                label=label,
                run_dir=run_dir,
                documents=documents,
                controlled_sources=controlled_sources,
                created_epoch=now,
            )
            self._jobs[job_id] = record
            self._fingerprints[fingerprint] = job_id
            self._futures[job_id] = self._executor.submit(self._execute, job_id)
            return record, False

    def _prepare_inputs(self, record: JobRecord) -> tuple[Path, Path, Path | None]:
        source_dir = record.run_dir / "inputs" / "sources"
        source_dir.mkdir(parents=True, exist_ok=True)
        source_paths = []
        for index, (title, text) in enumerate(record.documents, start=1):
            path = source_dir / f"source_{index}.txt"
            path.write_text(f"{title}\n\n{text}\n", encoding="utf-8")
            source_paths.append(path)

        benchmark = record.run_dir / "inputs" / "independent_benchmark.jsonl"
        benchmark.parent.mkdir(parents=True, exist_ok=True)
        benchmark_rows = (
            {
                "id": "weather-001",
                "question": "How are rainfall totals compared across remote mountain gauges?",
                "answer": "Calibrate gauges, align observation windows and compare accumulated depth.",
            },
            {
                "id": "astronomy-002",
                "question": "Why does a transit cause a star's measured brightness to fall?",
                "answer": "An orbiting body blocks a small portion of the observed light.",
            },
            {
                "id": "ecology-003",
                "question": "What can indicate recovery in a restored wetland?",
                "answer": "Species diversity, water quality and seasonal habitat persistence.",
            },
        )
        benchmark.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in benchmark_rows),
            encoding="utf-8",
        )

        source_manifest: Path | None = None
        if record.controlled_sources:
            source_manifest = record.run_dir / "inputs" / "source_rights.json"
            source_manifest.write_text(
                json.dumps(
                    [
                        {
                            "source_path": str(path),
                            "origin": "repository-controlled demonstration fixture",
                            "license": "MIT",
                            "permitted_uses": ["training", "evaluation"],
                            "rights_holder": "Forge demonstration",
                        }
                        for path in source_paths
                    ],
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
        return source_dir, benchmark, source_manifest

    def _execute(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = "running"
            record.started_epoch = time.time()
        try:
            source_dir, benchmark, source_manifest = self._prepare_inputs(record)
            run_dir = record.run_dir / "run"
            config = ForgeConfig.from_paths(
                sources=[str(source_dir)],
                benchmarks=[str(benchmark)],
                source_manifest=str(source_manifest) if source_manifest else None,
                tasks=("qa", "summary"),
                max_examples=24,
                max_chars=420,
                overlap=60,
                select_n=20,
                test_fraction=0.25,
                split_seed=42,
                pii_action="reject",
                dry_run=True,
                dataset_name="forge-hosted-demonstration",
                dataset_version="0.1.0",
                dataset_license="MIT" if record.controlled_sources else "NOASSERTION",
                benchmark_origin="repository-controlled independent smoke fixture",
                release_tier="smoke",
            )
            result = run_forge(run_dir, config, resume=False, cache_enabled=True)
            verification = verify_release(run_dir / "release_manifest.json")
            with self._lock:
                record.release_id = result.release_id
                record.verified_artifacts = int(verification["artifacts"])
                record.status = "succeeded"
                record.completed_epoch = time.time()
        except StageExecutionError as exc:
            with self._lock:
                record.status = "failed"
                record.public_error = f"The pipeline stopped at {exc.stage_name}. Review the stage evidence and try different source text."
                record.completed_epoch = time.time()
        except Exception:
            with self._lock:
                record.status = "failed"
                record.public_error = "Forge could not complete this bounded run. Please try again."
                record.completed_epoch = time.time()

    def _record(self, job_id: str) -> JobRecord:
        if len(job_id) != 16 or any(character not in "0123456789abcdef" for character in job_id):
            raise KeyError(job_id)
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            return record

    def _events(self, record: JobRecord) -> list[dict[str, Any]]:
        path = record.run_dir / "run" / "pipeline_events.jsonl"
        if not path.is_file():
            return []
        events: list[dict[str, Any]] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(value)
        return events

    def _stages(self, record: JobRecord) -> list[dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for event in self._events(record):
            stage = str(event.get("stage", ""))
            if stage in STAGE_ORDER:
                latest[stage] = event
        stages = []
        for name in STAGE_ORDER:
            event = latest.get(name, {})
            event_type = event.get("event_type")
            status = {
                "started": "running",
                "completed": "passed",
                "cache_hit": "passed",
                "failed": "failed",
            }.get(str(event_type), "pending")
            stages.append(
                {
                    "name": name,
                    "label": name.replace("_", " ").title(),
                    "status": status,
                    "execution": "cached" if event_type == "cache_hit" else "executed",
                    "elapsed_seconds": event.get("elapsed_seconds"),
                    "metrics": event.get("metrics", {}),
                }
            )
        return stages

    def _summary(self, record: JobRecord) -> dict[str, Any] | None:
        run_dir = record.run_dir / "run"
        release_path = run_dir / "release_manifest.json"
        if record.status != "succeeded" or not release_path.is_file():
            return None
        release = _read_json(release_path)
        split = _read_json(run_dir / "split_manifest.json")
        source_governance = _read_json(run_dir / "source_governance_report.json")
        record_governance = _read_json(run_dir / "record_governance_report.json")
        dedupe = _read_json(run_dir / "dedupe_report.json")
        contamination = _read_json(run_dir / "contamination_report.json")
        artifacts = release.get("artifacts", {})
        train = artifacts.get("train", {}) if isinstance(artifacts, dict) else {}
        test = artifacts.get("test", {}) if isinstance(artifacts, dict) else {}
        gates = release.get("gates", [])
        return {
            "release_id": release.get("release_id"),
            "release_tier": release.get("release_tier"),
            "verified": True,
            "verified_artifacts": record.verified_artifacts,
            "gates_passed": sum(1 for gate in gates if isinstance(gate, dict) and gate.get("status") == "passed"),
            "records": int(train.get("records", 0)) + int(test.get("records", 0)),
            "train_records": int(split.get("train", 0)),
            "test_records": int(split.get("test", 0)),
            "source_overlap": len(split.get("source_overlap", [])),
            "source_documents": int(source_governance.get("input_documents", 0)),
            "unknown_rights": int(source_governance.get("unknown_rights", 0)),
            "rejected_records": int(record_governance.get("rejected_records", 0)),
            "duplicates_removed": int(dedupe.get("dropped_examples", 0)),
            "contamination_flags": int(contamination.get("contaminated_count", 0)),
            "claim_status": release.get("claim_status", {}),
        }

    def status(self, job_id: str) -> dict[str, Any]:
        record = self._record(job_id)
        stages = self._stages(record)
        completed = sum(stage["status"] == "passed" for stage in stages)
        current = next((stage["name"] for stage in stages if stage["status"] == "running"), None)
        if current is None and record.status == "queued":
            current = "queued"
        elapsed = None
        if record.started_epoch is not None:
            elapsed = round((record.completed_epoch or time.time()) - record.started_epoch, 3)
        return {
            "job_id": record.job_id,
            "label": record.label,
            "status": record.status,
            "created_at": _iso(record.created_epoch),
            "started_at": _iso(record.started_epoch),
            "completed_at": _iso(record.completed_epoch),
            "elapsed_seconds": elapsed,
            "current_stage": current,
            "completed_stages": completed,
            "total_stages": len(STAGE_ORDER),
            "progress": round(completed / len(STAGE_ORDER), 4),
            "stages": stages,
            "error": record.public_error,
            "summary": self._summary(record),
        }

    def artifact_list(self, job_id: str) -> list[dict[str, Any]]:
        record = self._record(job_id)
        if record.status != "succeeded":
            raise HostedValidationError("Artifacts are available after the run succeeds.")
        run_dir = record.run_dir / "run"
        artifacts = []
        for key, (filename, label, media_type) in PUBLIC_ARTIFACTS.items():
            path = run_dir / filename
            if path.is_file():
                artifacts.append(
                    {
                        "key": key,
                        "filename": filename,
                        "label": label,
                        "media_type": media_type,
                        "bytes": path.stat().st_size,
                        "records": _count_jsonl(path) if filename.endswith(".jsonl") else None,
                    }
                )
        return artifacts

    def artifact(self, job_id: str, key: str) -> tuple[Path, str]:
        record = self._record(job_id)
        if record.status != "succeeded":
            raise HostedValidationError("Artifacts are available after the run succeeds.")
        artifact = PUBLIC_ARTIFACTS.get(key)
        if artifact is None:
            raise KeyError(key)
        filename, _, media_type = artifact
        path = record.run_dir / "run" / filename
        if not path.is_file():
            raise KeyError(key)
        return path, media_type

    def evidence_bundle(self, job_id: str) -> bytes:
        record = self._record(job_id)
        if record.status != "succeeded":
            raise HostedValidationError("The evidence bundle is available after the run succeeds.")
        run_dir = record.run_dir / "run"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "README.txt",
                "Forge deterministic smoke evidence\n\n"
                "This bundle proves that the declared pipeline controls executed and the release verified.\n"
                "It does not establish model quality or replace independent human evaluation.\n",
            )
            for filename, _, _ in PUBLIC_ARTIFACTS.values():
                path = run_dir / filename
                if path.is_file():
                    archive.write(path, arcname=filename)
        return buffer.getvalue()

    def wait(self, job_id: str, timeout: float = 20.0) -> dict[str, Any]:
        future = self._futures.get(job_id)
        if future is not None:
            future.result(timeout=timeout)
        return self.status(job_id)
