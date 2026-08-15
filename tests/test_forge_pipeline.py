"""Integrity and recovery tests for Forge's canonical execution engine."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from forge import (
    ArtifactBinding,
    ModelRef,
    Pipeline,
    PromptRef,
    StageContext,
    StageDefinition,
    StageExecutionError,
)
from forge.contracts import canonical_sha256, fingerprint_path


@dataclass(frozen=True)
class WriteConfig:
    output: str
    content: str


def _write(context: StageContext, config: WriteConfig) -> dict[str, int]:
    context.path(config.output).write_text(config.content, encoding="utf-8")
    return {"characters": len(config.content)}


def _copy(context: StageContext, config: WriteConfig) -> dict[str, int]:
    source = context.path("first.txt").read_text(encoding="utf-8")
    context.path(config.output).write_text(source + config.content, encoding="utf-8")
    return {"characters": len(source + config.content)}


def _stage(name: str, config: WriteConfig, *, depends_on: tuple[str, ...] = ()) -> StageDefinition[WriteConfig]:
    inputs = (ArtifactBinding("first", "first.txt", media_type="text/plain"),) if depends_on else ()
    return StageDefinition(
        name=name,
        version="1",
        config=config,
        runner=_copy if depends_on else _write,
        inputs=inputs,
        outputs=(ArtifactBinding(name, config.output, media_type="text/plain"),),
        depends_on=depends_on,
    )


def test_identical_run_uses_verified_content_cache(tmp_path: Path) -> None:
    first = Pipeline(tmp_path)
    first.add(_stage("first", WriteConfig("first.txt", "alpha")))
    first.add(_stage("second", WriteConfig("second.txt", " beta"), depends_on=("first",)))
    initial = first.run()

    resumed = Pipeline(tmp_path)
    resumed.add(_stage("first", WriteConfig("first.txt", "alpha")))
    resumed.add(_stage("second", WriteConfig("second.txt", " beta"), depends_on=("first",)))
    repeated = resumed.run()

    assert [result.status for result in initial] == ["ok", "ok"]
    assert [result.status for result in repeated] == ["cached", "cached"]
    assert [result.cache_key for result in repeated] == [result.cache_key for result in initial]
    assert (tmp_path / "second.txt").read_text(encoding="utf-8") == "alpha beta"


def test_tampered_output_invalidates_only_affected_stage(tmp_path: Path) -> None:
    first = Pipeline(tmp_path)
    first.add(_stage("first", WriteConfig("first.txt", "alpha")))
    first.add(_stage("second", WriteConfig("second.txt", " beta"), depends_on=("first",)))
    first.run()
    (tmp_path / "second.txt").write_text("tampered", encoding="utf-8")

    resumed = Pipeline(tmp_path)
    resumed.add(_stage("first", WriteConfig("first.txt", "alpha")))
    resumed.add(_stage("second", WriteConfig("second.txt", " beta"), depends_on=("first",)))
    results = resumed.run()

    assert [result.status for result in results] == ["cached", "ok"]
    assert (tmp_path / "second.txt").read_text(encoding="utf-8") == "alpha beta"


def test_configuration_change_invalidates_stage_cache(tmp_path: Path) -> None:
    initial = Pipeline(tmp_path)
    initial.add(_stage("first", WriteConfig("first.txt", "alpha")))
    first_key = initial.run()[0].cache_key

    changed = Pipeline(tmp_path)
    changed.add(_stage("first", WriteConfig("first.txt", "bravo")))
    result = changed.run()[0]

    assert result.status == "ok"
    assert result.cache_key != first_key
    assert (tmp_path / "first.txt").read_text(encoding="utf-8") == "bravo"


def test_failed_stage_resumes_after_last_verified_artifact(tmp_path: Path) -> None:
    attempts = {"count": 0}

    def fail_once(context: StageContext, config: WriteConfig) -> dict[str, int]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("injected failure")
        return _copy(context, config)

    initial = Pipeline(tmp_path)
    initial.add(_stage("first", WriteConfig("first.txt", "alpha")))
    initial.add(
        StageDefinition(
            name="second",
            version="1",
            config=WriteConfig("second.txt", " recovered"),
            runner=fail_once,
            inputs=(ArtifactBinding("first", "first.txt", media_type="text/plain"),),
            outputs=(ArtifactBinding("second", "second.txt", media_type="text/plain"),),
            depends_on=("first",),
        )
    )
    with pytest.raises(StageExecutionError, match="injected failure"):
        initial.run()

    resumed = Pipeline(tmp_path)
    resumed.add(_stage("first", WriteConfig("first.txt", "alpha")))
    resumed.add(
        StageDefinition(
            name="second",
            version="1",
            config=WriteConfig("second.txt", " recovered"),
            runner=fail_once,
            inputs=(ArtifactBinding("first", "first.txt", media_type="text/plain"),),
            outputs=(ArtifactBinding("second", "second.txt", media_type="text/plain"),),
            depends_on=("first",),
        )
    )
    results = resumed.run()

    assert [result.status for result in results] == ["cached", "ok"]
    assert (tmp_path / "second.txt").read_text(encoding="utf-8") == "alpha recovered"
    events = [json.loads(line) for line in (tmp_path / "pipeline_events.jsonl").read_text().splitlines()]
    assert any(event["event_type"] == "failed" and event["stage"] == "second" for event in events)
    assert any(event["event_type"] == "completed" and event["stage"] == "second" for event in events)


def test_events_capture_contract_model_prompt_and_artifacts(tmp_path: Path) -> None:
    prompt = PromptRef("qa", "v1", canonical_sha256({"prompt": "question"}))
    pipeline = Pipeline(tmp_path)
    pipeline.add(
        StageDefinition(
            name="generate",
            version="7",
            config=WriteConfig("out.txt", "evidence"),
            runner=_write,
            outputs=(ArtifactBinding("examples", "out.txt", media_type="text/plain"),),
            models=(ModelRef("test", "deterministic", revision="1"),),
            prompts=(prompt,),
        )
    )
    pipeline.run()

    events = [json.loads(line) for line in (tmp_path / "pipeline_events.jsonl").read_text().splitlines()]
    completed = next(event for event in events if event["event_type"] == "completed")
    assert completed["schema_version"] == "forge.event/v1"
    assert completed["stage_version"] == "7"
    assert completed["models"][0]["name"] == "deterministic"
    assert completed["prompts"][0]["sha256"] == prompt.sha256
    assert completed["outputs"][0]["sha256"] == fingerprint_path(tmp_path / "out.txt").sha256
    assert "content" in completed["config"]


def test_external_directory_fingerprint_ignores_absolute_location(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a.txt").write_text("same", encoding="utf-8")
    (second / "a.txt").write_text("same", encoding="utf-8")
    assert fingerprint_path(first).sha256 == fingerprint_path(second).sha256


def test_installed_forge_command_exposes_canonical_cli() -> None:
    result = subprocess.run(["forge", "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "verifiable training-data curation pipeline" in result.stdout
