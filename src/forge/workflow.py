"""The single public API for constructing and running a Forge curation workflow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training_data_robo.cli import build_task_templates_from_names
from training_data_robo.releases import verify_release, write_release

from .contracts import (
    ArtifactBinding,
    ContaminationConfig,
    DedupeConfig,
    DifficultyConfig,
    GenerationConfig,
    IngestConfig,
    JudgmentConfig,
    ModelRef,
    ProfileConfig,
    PromptRef,
    QualityConfig,
    RecordGovernanceConfig,
    SelectionConfig,
    SourceGovernanceConfig,
    SplitConfig,
    canonical_sha256,
)
from .pipeline import Pipeline, StageDefinition, StageResult
from .stages import (
    run_contamination,
    run_dedupe,
    run_difficulty,
    run_generation,
    run_ingest,
    run_judgment,
    run_profile,
    run_quality,
    run_record_governance,
    run_selection,
    run_source_governance,
    run_split,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ForgeConfig:
    sources: tuple[str, ...]
    benchmarks: tuple[str, ...]
    source_manifest: str | None = None
    tasks: tuple[str, ...] = ("qa", "summary")
    model: str = "gpt-4.1-mini"
    judge_model: str = "gpt-4.1-mini"
    max_examples: int = 200
    max_chars: int = 900
    overlap: int = 100
    select_n: int | None = None
    select_strategy: str = "quality_weighted"
    test_fraction: float = 0.2
    split_seed: int = 42
    pii_action: str = "reject"
    fuzzy_dedupe_threshold: float = 0.8
    fuzzy_contamination_threshold: float = 0.8
    semantic_backend: str = "disabled"
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_revision: str = "main"
    semantic_dedupe_threshold: float = 0.9
    semantic_contamination_threshold: float = 0.9
    dry_run: bool = False
    dataset_name: str = "forge-generated-dataset"
    dataset_version: str = "0.1.0"
    dataset_license: str = "NOASSERTION"
    benchmark_origin: str = "repository synthetic contamination fixture"
    release_tier: str = "smoke"

    def validate(self) -> None:
        if not self.sources:
            raise ValueError("Forge requires at least one source")
        if not self.benchmarks:
            raise ValueError("Forge requires at least one independent benchmark")
        if not self.tasks:
            raise ValueError("Forge requires at least one generation task")
        if self.max_examples < 1:
            raise ValueError("max_examples must be positive")
        if self.max_chars < 100:
            raise ValueError("max_chars must be at least 100")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction must be between 0 and 1")
        if self.release_tier not in {"smoke", "candidate"}:
            raise ValueError("release_tier must be 'smoke' or 'candidate'")
        if self.pii_action not in {"reject", "redact"}:
            raise ValueError("pii_action must be 'reject' or 'redact'")
        if self.semantic_backend not in {"disabled", "sentence_transformers", "openai"}:
            raise ValueError("semantic_backend must be disabled, sentence_transformers, or openai")
        for field_name in (
            "fuzzy_dedupe_threshold",
            "fuzzy_contamination_threshold",
            "semantic_dedupe_threshold",
            "semantic_contamination_threshold",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between zero and one")
        if self.release_tier == "candidate" and not self.source_manifest:
            raise ValueError("candidate releases require a source-rights manifest")
        if self.release_tier == "candidate" and self.semantic_backend == "disabled":
            raise ValueError("candidate releases require a semantic dedupe and contamination backend")
        for field_name in ("dataset_name", "dataset_version", "dataset_license", "benchmark_origin"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must not be empty")

    @classmethod
    def from_paths(cls, *, sources: list[str], benchmarks: list[str], **kwargs: Any) -> "ForgeConfig":
        resolved_sources = tuple(str(Path(value).expanduser().resolve()) for value in sources)
        resolved_benchmarks = tuple(str(Path(value).expanduser().resolve()) for value in benchmarks)
        if kwargs.get("source_manifest"):
            kwargs["source_manifest"] = str(Path(str(kwargs["source_manifest"])).expanduser().resolve())
        return cls(sources=resolved_sources, benchmarks=resolved_benchmarks, **kwargs)


@dataclass(frozen=True)
class ForgeRun:
    run_dir: Path
    release_id: str
    release_tier: str
    stage_results: tuple[StageResult, ...]

    @property
    def cache_hits(self) -> int:
        return sum(result.status == "cached" for result in self.stage_results)


def _json_binding(role: str, path: str, *, scope: str = "run") -> ArtifactBinding:
    media_type = "application/x-ndjson" if path.endswith(".jsonl") else "application/json"
    return ArtifactBinding(role=role, path=path, scope=scope, media_type=media_type)


def _generation_evidence(config: ForgeConfig) -> tuple[tuple[ModelRef, ...], tuple[PromptRef, ...]]:
    templates = build_task_templates_from_names(list(config.tasks))
    prompts = tuple(
        PromptRef(
            name=template.name,
            version=template.version,
            sha256=canonical_sha256(
                {
                    "system_prompt": template.system_prompt,
                    "user_prompt_template": template.user_prompt_template,
                    "temperature": template.temperature,
                    "top_p": template.top_p,
                    "max_output_tokens": template.max_output_tokens,
                }
            ),
        )
        for template in templates
    )
    model = ModelRef(
        provider="forge" if config.dry_run else "openai",
        name="deterministic-dummy" if config.dry_run else config.model,
        parameters={"mode": "smoke" if config.dry_run else "live"},
    )
    return (model,), prompts


def build_pipeline(run_dir: Path, config: ForgeConfig, *, cache_enabled: bool = True) -> Pipeline:
    """Build the canonical Forge stage graph without executing it."""
    config.validate()
    pipeline = Pipeline(run_dir, cache_enabled=cache_enabled)
    generation_models, generation_prompts = _generation_evidence(config)
    judge_model = ModelRef(
        provider="forge" if config.dry_run else "openai",
        name="deterministic-dummy-judge" if config.dry_run else config.judge_model,
        parameters={"rubric": "forge-default-v1", "max_concurrent": 5},
    )
    semantic_models = (
        (
            ModelRef(
                provider=config.semantic_backend,
                name=config.semantic_model,
                revision=config.semantic_revision,
                parameters={"purpose": "similarity_control"},
            ),
        )
        if config.semantic_backend != "disabled"
        else ()
    )

    pipeline.add(
        StageDefinition(
            name="ingest",
            version="1",
            config=IngestConfig(sources=config.sources),
            runner=run_ingest,
            inputs=tuple(
                ArtifactBinding(role=f"source_{index}", path=source, scope="external")
                for index, source in enumerate(config.sources, start=1)
            ),
            outputs=(_json_binding("documents", "documents.jsonl"),),
        )
    )
    pipeline.add(
        StageDefinition(
            name="source_governance",
            version="1",
            config=SourceGovernanceConfig(
                source_manifest=config.source_manifest,
                pii_action=config.pii_action,
                allow_unknown_rights=config.release_tier == "smoke",
            ),
            runner=run_source_governance,
            inputs=(
                _json_binding("documents", "documents.jsonl"),
                *(
                    (
                        ArtifactBinding(
                            role="source_rights_manifest",
                            path=config.source_manifest,
                            scope="external",
                            media_type="application/json",
                        ),
                    )
                    if config.source_manifest
                    else ()
                ),
            ),
            outputs=(
                _json_binding("governed_documents", "governed_documents.jsonl"),
                _json_binding("rejected_documents", "rejected_documents.jsonl"),
                _json_binding("source_governance_report", "source_governance_report.json"),
            ),
            depends_on=("ingest",),
        )
    )
    pipeline.add(
        StageDefinition(
            name="generate",
            version="2",
            config=GenerationConfig(
                tasks=config.tasks,
                model=config.model,
                max_examples=config.max_examples,
                max_chars=config.max_chars,
                overlap=config.overlap,
                dry_run=config.dry_run,
            ),
            runner=run_generation,
            inputs=(_json_binding("governed_documents", "governed_documents.jsonl"),),
            outputs=(_json_binding("raw_examples", "raw_dataset.jsonl"),),
            depends_on=("source_governance",),
            models=generation_models,
            prompts=generation_prompts,
        )
    )
    pipeline.add(
        StageDefinition(
            name="quality",
            version="1",
            config=QualityConfig(),
            runner=run_quality,
            inputs=(_json_binding("raw_examples", "raw_dataset.jsonl"),),
            outputs=(_json_binding("scored_examples", "quality.jsonl"),),
            depends_on=("generate",),
        )
    )
    pipeline.add(
        StageDefinition(
            name="record_governance",
            version="1",
            config=RecordGovernanceConfig(pii_action=config.pii_action),
            runner=run_record_governance,
            inputs=(_json_binding("scored_examples", "quality.jsonl"),),
            outputs=(
                _json_binding("governed_records", "governed_records.jsonl"),
                _json_binding("rejected_records", "rejected_records.jsonl"),
                _json_binding("record_governance_report", "record_governance_report.json"),
            ),
            depends_on=("quality",),
        )
    )
    pipeline.add(
        StageDefinition(
            name="dedupe",
            version="2",
            config=DedupeConfig(
                fuzzy_threshold=config.fuzzy_dedupe_threshold,
                semantic_backend=config.semantic_backend,
                semantic_model=config.semantic_model,
                semantic_revision=config.semantic_revision,
                semantic_threshold=config.semantic_dedupe_threshold,
            ),
            runner=run_dedupe,
            inputs=(_json_binding("governed_records", "governed_records.jsonl"),),
            outputs=(
                _json_binding("unique_examples", "deduped.jsonl"),
                _json_binding("duplicate_examples", "dedupe_rejections.jsonl"),
                _json_binding("dedupe_report", "dedupe_report.json"),
            ),
            depends_on=("record_governance",),
            models=semantic_models,
        )
    )
    pipeline.add(
        StageDefinition(
            name="judge",
            version="1",
            config=JudgmentConfig(model=config.judge_model, dry_run=config.dry_run),
            runner=run_judgment,
            inputs=(_json_binding("unique_examples", "deduped.jsonl"),),
            outputs=(_json_binding("judged_examples", "judged.jsonl"),),
            depends_on=("dedupe",),
            models=(judge_model,),
        )
    )
    contamination_inputs = [_json_binding("judged_examples", "judged.jsonl")]
    contamination_inputs.extend(
        ArtifactBinding(
            role=f"benchmark_{index}",
            path=benchmark,
            scope="external",
            media_type="application/x-ndjson",
        )
        for index, benchmark in enumerate(config.benchmarks, start=1)
    )
    pipeline.add(
        StageDefinition(
            name="contamination",
            version="2",
            config=ContaminationConfig(
                benchmark_paths=config.benchmarks,
                fuzzy_threshold=config.fuzzy_contamination_threshold,
                semantic_backend=config.semantic_backend,
                semantic_model=config.semantic_model,
                semantic_revision=config.semantic_revision,
                semantic_threshold=config.semantic_contamination_threshold,
            ),
            runner=run_contamination,
            inputs=tuple(contamination_inputs),
            outputs=(_json_binding("contamination_report", "contamination_report.json"),),
            depends_on=("judge",),
            models=semantic_models,
        )
    )
    pipeline.add(
        StageDefinition(
            name="difficulty",
            version="1",
            config=DifficultyConfig(),
            runner=run_difficulty,
            inputs=(_json_binding("judged_examples", "judged.jsonl"),),
            outputs=(_json_binding("calibrated_examples", "difficulty.jsonl"),),
            depends_on=("contamination",),
        )
    )
    pipeline.add(
        StageDefinition(
            name="select",
            version="1",
            config=SelectionConfig(count=config.select_n, strategy=config.select_strategy),
            runner=run_selection,
            inputs=(_json_binding("calibrated_examples", "difficulty.jsonl"),),
            outputs=(_json_binding("selected_examples", "selected.jsonl"),),
            depends_on=("difficulty",),
        )
    )
    pipeline.add(
        StageDefinition(
            name="split",
            version="1",
            config=SplitConfig(test_fraction=config.test_fraction, seed=config.split_seed),
            runner=run_split,
            inputs=(_json_binding("selected_examples", "selected.jsonl"),),
            outputs=(
                _json_binding("train_split", "train.jsonl"),
                _json_binding("test_split", "test.jsonl"),
                _json_binding("split_manifest", "split_manifest.json"),
            ),
            depends_on=("select",),
        )
    )
    profile_inputs = (
        _json_binding("train_split", "train.jsonl"),
        _json_binding("test_split", "test.jsonl"),
        _json_binding("source_governance_report", "source_governance_report.json"),
        _json_binding("record_governance_report", "record_governance_report.json"),
        _json_binding("dedupe_report", "dedupe_report.json"),
        _json_binding("contamination_report", "contamination_report.json"),
        _json_binding("rejected_documents", "rejected_documents.jsonl"),
        _json_binding("rejected_records", "rejected_records.jsonl"),
        _json_binding("dedupe_rejections", "dedupe_rejections.jsonl"),
    )
    pipeline.add(
        StageDefinition(
            name="profile",
            version="1",
            config=ProfileConfig(),
            runner=run_profile,
            inputs=profile_inputs,
            outputs=(_json_binding("dataset_profile", "dataset_profile.json"),),
            depends_on=("split",),
        )
    )
    return pipeline


def _write_run_config(run_dir: Path, config: ForgeConfig, started_at: str) -> None:
    payload = asdict(config)
    public_sources = [Path(value).name for value in config.sources]
    public_benchmarks = [Path(value).name for value in config.benchmarks]
    payload["sources"] = public_sources
    payload["benchmarks"] = public_benchmarks
    payload["source"] = public_sources
    payload["benchmark_file"] = public_benchmarks
    payload["source_manifest"] = Path(config.source_manifest).name if config.source_manifest else None
    payload["started_at"] = started_at
    payload["pipeline_api"] = "forge.workflow/v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = run_dir / ".forge"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "runtime_inputs.json").write_text(
        json.dumps(
            {
                "sources": list(config.sources),
                "benchmarks": list(config.benchmarks),
                "source_manifest": config.source_manifest,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def run_forge(
    run_dir: Path,
    config: ForgeConfig,
    *,
    resume: bool = True,
    cache_enabled: bool = True,
) -> ForgeRun:
    """Run Forge from any CLI, test, worker, or future HTTP service."""
    config.validate()
    run_dir = run_dir.expanduser().resolve()
    started_at = _now()
    _write_run_config(run_dir, config, started_at)
    pipeline = build_pipeline(run_dir, config, cache_enabled=cache_enabled)
    results = tuple(pipeline.run(resume=resume))

    run_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    run_config["completed_at"] = _now()
    run_config["stage_cache_hits"] = sum(result.status == "cached" for result in results)
    (run_dir / "config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    release = write_release(
        run_dir,
        dataset_name=config.dataset_name,
        dataset_version=config.dataset_version,
        dataset_license=config.dataset_license,
        benchmark_origin=config.benchmark_origin,
        release_tier=config.release_tier,
    )
    verify_release(run_dir / "release_manifest.json")
    return ForgeRun(
        run_dir=run_dir,
        release_id=str(release["release_id"]),
        release_tier=str(release["release_tier"]),
        stage_results=results,
    )
