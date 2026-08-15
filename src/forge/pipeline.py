"""Content-keyed, resumable execution engine used by every Forge entry point."""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Generic, Mapping, TypeVar

from .contracts import (
    ArtifactBinding,
    ArtifactRef,
    JsonValue,
    ModelRef,
    PromptRef,
    canonical_sha256,
    json_value,
)

ConfigT = TypeVar("ConfigT")
StageRunner = Callable[["StageContext", ConfigT], Mapping[str, JsonValue]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(json_value(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class StageContext:
    run_dir: Path
    run_id: str
    stage_name: str
    cache_key: str

    def path(self, relative_path: str) -> Path:
        return self.run_dir / relative_path


@dataclass(frozen=True)
class StageDefinition(Generic[ConfigT]):
    """A typed stage plus its declared evidence boundary."""

    name: str
    version: str
    config: ConfigT
    runner: StageRunner[ConfigT]
    inputs: tuple[ArtifactBinding, ...] = ()
    outputs: tuple[ArtifactBinding, ...] = ()
    depends_on: tuple[str, ...] = ()
    models: tuple[ModelRef, ...] = ()
    prompts: tuple[PromptRef, ...] = ()

    def public_contract(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "version": self.version,
            "depends_on": list(self.depends_on),
            "config": json_value(self.config),
            "inputs": [json_value(binding) for binding in self.inputs],
            "outputs": [json_value(binding) for binding in self.outputs],
            "models": [json_value(model) for model in self.models],
            "prompts": [json_value(prompt) for prompt in self.prompts],
        }


@dataclass(frozen=True)
class StageResult:
    name: str
    version: str
    status: str
    cache_key: str
    elapsed_seconds: float
    inputs: tuple[ArtifactRef, ...]
    outputs: tuple[ArtifactRef, ...]
    metrics: Mapping[str, JsonValue]
    error: str | None = None


class StageExecutionError(RuntimeError):
    def __init__(self, stage_name: str, message: str) -> None:
        self.stage_name = stage_name
        super().__init__(f"Forge stage {stage_name!r} failed: {message}")


class Pipeline:
    """Execute a stage DAG with verified content cache and structured events."""

    def __init__(self, run_dir: Path, *, cache_enabled: bool = True) -> None:
        self.run_dir = run_dir.resolve()
        self.cache_enabled = cache_enabled
        self._stages: dict[str, StageDefinition[Any]] = {}
        self._results: list[StageResult] = []
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def state_path(self) -> Path:
        return self.run_dir / ".forge" / "state.json"

    @property
    def events_path(self) -> Path:
        return self.run_dir / "pipeline_events.jsonl"

    @property
    def log_path(self) -> Path:
        return self.run_dir / "pipeline_log.json"

    def cache_path(self, stage_name: str) -> Path:
        safe_name = stage_name.replace("/", "_")
        return self.run_dir / ".forge" / "cache" / f"{safe_name}.json"

    def add(self, stage: StageDefinition[Any]) -> None:
        if stage.name in self._stages:
            raise ValueError(f"Stage {stage.name!r} is already registered")
        missing = [dependency for dependency in stage.depends_on if dependency not in self._stages]
        if missing:
            raise ValueError(f"Stage {stage.name!r} has unregistered dependencies: {', '.join(missing)}")
        self._stages[stage.name] = stage

    def _topological_order(self) -> list[str]:
        in_degree = {name: 0 for name in self._stages}
        adjacency: dict[str, list[str]] = {name: [] for name in self._stages}
        for name, stage in self._stages.items():
            for dependency in stage.depends_on:
                adjacency[dependency].append(name)
                in_degree[name] += 1

        queue: deque[str] = deque(name for name, degree in in_degree.items() if degree == 0)
        order: list[str] = []
        while queue:
            current = queue.popleft()
            order.append(current)
            for child in adjacency[current]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        if len(order) != len(self._stages):
            raise ValueError("Forge pipeline contains a dependency cycle")
        return order

    def _run_id(self, order: list[str]) -> str:
        state = self._load_json(self.state_path)
        existing = state.get("run_id")
        if isinstance(existing, str) and existing:
            return existing
        plan = [self._stages[name].public_contract() for name in order]
        return f"forge_run_{canonical_sha256(plan)[:20]}"

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return value if isinstance(value, dict) else {}

    def _capture(self, bindings: tuple[ArtifactBinding, ...]) -> tuple[ArtifactRef, ...]:
        return tuple(ArtifactRef.capture(binding, self.run_dir) for binding in bindings)

    def _cache_key(self, stage: StageDefinition[Any], inputs: tuple[ArtifactRef, ...]) -> str:
        identity = {
            "stage": stage.public_contract(),
            "inputs": [json_value(artifact) for artifact in inputs],
        }
        return canonical_sha256(identity)

    def _load_cached_result(
        self,
        stage: StageDefinition[Any],
        cache_key: str,
        inputs: tuple[ArtifactRef, ...],
    ) -> StageResult | None:
        cached = self._load_json(self.cache_path(stage.name))
        if cached.get("cache_key") != cache_key:
            return None
        output_values = cached.get("outputs")
        if not isinstance(output_values, list):
            return None
        try:
            outputs = tuple(ArtifactRef(**item) for item in output_values if isinstance(item, dict))
        except TypeError:
            return None
        if len(outputs) != len(stage.outputs) or not all(output.verify(self.run_dir) for output in outputs):
            return None
        metrics = cached.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        return StageResult(
            name=stage.name,
            version=stage.version,
            status="cached",
            cache_key=cache_key,
            elapsed_seconds=0.0,
            inputs=inputs,
            outputs=outputs,
            metrics=metrics,
        )

    def _append_event(self, event: Mapping[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_value(event), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _event(
        self,
        *,
        run_id: str,
        stage: StageDefinition[Any],
        event_type: str,
        cache_key: str,
        inputs: tuple[ArtifactRef, ...],
        outputs: tuple[ArtifactRef, ...] = (),
        metrics: Mapping[str, JsonValue] | None = None,
        error: str | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "schema_version": "forge.event/v1",
            "event_id": canonical_sha256(
                {
                    "run_id": run_id,
                    "stage": stage.name,
                    "event_type": event_type,
                    "cache_key": cache_key,
                    "timestamp": _utc_now(),
                }
            ),
            "event_type": event_type,
            "timestamp": _utc_now(),
            "run_id": run_id,
            "stage": stage.name,
            "stage_version": stage.version,
            "cache_key": cache_key,
            "config": json_value(stage.config),
            "models": [json_value(model) for model in stage.models],
            "prompts": [json_value(prompt) for prompt in stage.prompts],
            "inputs": [json_value(artifact) for artifact in inputs],
            "outputs": [json_value(artifact) for artifact in outputs],
        }
        if metrics is not None:
            payload["metrics"] = json_value(metrics)
        if error is not None:
            payload["error"] = error
        if elapsed_seconds is not None:
            payload["elapsed_seconds"] = elapsed_seconds
        self._append_event(payload)

    def _persist_state(self, run_id: str, order: list[str]) -> None:
        _atomic_json(
            self.state_path,
            {
                "schema_version": "forge.pipeline-state/v1",
                "run_id": run_id,
                "updated_at": _utc_now(),
                "plan": order,
                "completed": {
                    result.name: {
                        "cache_key": result.cache_key,
                        "status": result.status,
                        "outputs": [json_value(output) for output in result.outputs],
                    }
                    for result in self._results
                    if result.status in {"ok", "cached"}
                },
            },
        )

    def _persist_legacy_log(self) -> None:
        _atomic_json(
            self.log_path,
            [
                {
                    "step": result.name,
                    "status": "ok" if result.status in {"ok", "cached"} else "failed",
                    "execution": result.status,
                    "elapsed_seconds": result.elapsed_seconds,
                    "cache_key": result.cache_key,
                    "error": result.error,
                    "timestamp": _utc_now(),
                }
                for result in self._results
            ],
        )

    def run(self, *, resume: bool = True) -> list[StageResult]:
        order = self._topological_order()
        run_id = self._run_id(order)
        self._results = []

        for stage_name in order:
            stage = self._stages[stage_name]
            try:
                inputs = self._capture(stage.inputs)
            except Exception as exc:
                raise StageExecutionError(stage.name, str(exc)) from exc
            cache_key = self._cache_key(stage, inputs)

            cached = self._load_cached_result(stage, cache_key, inputs) if self.cache_enabled and resume else None
            if cached is not None:
                self._results.append(cached)
                self._event(
                    run_id=run_id,
                    stage=stage,
                    event_type="cache_hit",
                    cache_key=cache_key,
                    inputs=inputs,
                    outputs=cached.outputs,
                    metrics=cached.metrics,
                )
                self._persist_state(run_id, order)
                self._persist_legacy_log()
                continue

            self._event(
                run_id=run_id,
                stage=stage,
                event_type="started",
                cache_key=cache_key,
                inputs=inputs,
            )
            started = time.monotonic()
            context = StageContext(self.run_dir, run_id, stage.name, cache_key)
            try:
                metrics = dict(stage.runner(context, stage.config))
                outputs = self._capture(stage.outputs)
                elapsed = round(time.monotonic() - started, 6)
                result = StageResult(
                    name=stage.name,
                    version=stage.version,
                    status="ok",
                    cache_key=cache_key,
                    elapsed_seconds=elapsed,
                    inputs=inputs,
                    outputs=outputs,
                    metrics=metrics,
                )
                _atomic_json(
                    self.cache_path(stage.name),
                    {
                        "schema_version": "forge.stage-cache/v1",
                        "stage": stage.name,
                        "stage_version": stage.version,
                        "cache_key": cache_key,
                        "created_at": _utc_now(),
                        "inputs": [json_value(artifact) for artifact in inputs],
                        "outputs": [json_value(artifact) for artifact in outputs],
                        "metrics": metrics,
                    },
                )
                self._results.append(result)
                self._event(
                    run_id=run_id,
                    stage=stage,
                    event_type="completed",
                    cache_key=cache_key,
                    inputs=inputs,
                    outputs=outputs,
                    metrics=metrics,
                    elapsed_seconds=elapsed,
                )
                self._persist_state(run_id, order)
                self._persist_legacy_log()
            except Exception as exc:
                elapsed = round(time.monotonic() - started, 6)
                failed = StageResult(
                    name=stage.name,
                    version=stage.version,
                    status="failed",
                    cache_key=cache_key,
                    elapsed_seconds=elapsed,
                    inputs=inputs,
                    outputs=(),
                    metrics={},
                    error=str(exc),
                )
                self._results.append(failed)
                self._event(
                    run_id=run_id,
                    stage=stage,
                    event_type="failed",
                    cache_key=cache_key,
                    inputs=inputs,
                    error=str(exc),
                    elapsed_seconds=elapsed,
                )
                self._persist_state(run_id, order)
                self._persist_legacy_log()
                raise StageExecutionError(stage.name, str(exc)) from exc

        return list(self._results)

    def results(self) -> list[StageResult]:
        return list(self._results)
