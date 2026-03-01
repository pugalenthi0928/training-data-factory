"""DAG-based pipeline runner with dependency resolution, caching, and resume.

Provides a Pipeline class that executes processing steps in topological order,
skips steps whose outputs are already up-to-date, and supports resume from
the last successful step on failure.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .logging_config import get_logger

logger = get_logger("training_data_robo.pipeline")


@dataclass
class StepResult:
    """Result of a single pipeline step execution."""

    name: str
    status: str  # "ok", "failed", "skipped", "cached"
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    output_path: Optional[str] = None
    timestamp: str = ""


@dataclass
class PipelineStep:
    """A single step in the pipeline DAG."""

    name: str
    fn: Callable[..., Any]
    depends_on: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """DAG pipeline runner with topological ordering, caching, and resume.

    Usage::

        pipe = Pipeline(output_dir=Path("runs/001"))
        pipe.add_step("generate", generate_fn, outputs=["raw.jsonl"])
        pipe.add_step("quality", quality_fn, depends_on=["generate"],
                       inputs=["raw.jsonl"], outputs=["quality.jsonl"])
        results = pipe.run()
    """

    def __init__(self, output_dir: Path, cache_enabled: bool = True) -> None:
        self.output_dir = output_dir
        self.cache_enabled = cache_enabled
        self._steps: Dict[str, PipelineStep] = {}
        self._results: List[StepResult] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def log_path(self) -> Path:
        return self.output_dir / "pipeline_log.json"

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / ".pipeline_checkpoint.json"

    def add_step(
        self,
        name: str,
        fn: Callable[..., Any],
        depends_on: Optional[List[str]] = None,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Register a pipeline step."""
        if name in self._steps:
            raise ValueError(f"Step {name!r} already registered")

        # Validate dependencies exist
        for dep in depends_on or []:
            if dep not in self._steps:
                raise ValueError(
                    f"Step {name!r} depends on {dep!r}, which hasn't been added yet"
                )

        self._steps[name] = PipelineStep(
            name=name,
            fn=fn,
            depends_on=depends_on or [],
            outputs=outputs or [],
            inputs=inputs or [],
            kwargs=kwargs,
        )

    def _topological_sort(self) -> List[str]:
        """Topological sort of steps using Kahn's algorithm."""
        in_degree: Dict[str, int] = {name: 0 for name in self._steps}
        adjacency: Dict[str, List[str]] = {name: [] for name in self._steps}

        for name, step in self._steps.items():
            for dep in step.depends_on:
                adjacency[dep].append(name)
                in_degree[name] += 1

        queue: deque[str] = deque(
            name for name, deg in in_degree.items() if deg == 0
        )
        order: List[str] = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in adjacency[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self._steps):
            raise ValueError("Pipeline has circular dependencies")

        return order

    def _compute_input_hash(self, step: PipelineStep) -> str:
        """Compute a hash of input files for cache invalidation."""
        hasher = hashlib.sha256()
        for input_file in sorted(step.inputs):
            path = self.output_dir / input_file
            if path.exists():
                hasher.update(path.read_bytes())
            else:
                hasher.update(b"__missing__")
        return hasher.hexdigest()[:16]

    def _is_cached(self, step: PipelineStep) -> bool:
        """Check if step outputs exist and inputs haven't changed."""
        if not self.cache_enabled:
            return False

        if not step.outputs:
            return False

        # All outputs must exist
        for out_file in step.outputs:
            if not (self.output_dir / out_file).exists():
                return False

        # Check input hash against stored hash
        cache_file = self.output_dir / f".cache_{step.name}.json"
        if not cache_file.exists():
            return False

        try:
            stored = json.loads(cache_file.read_text(encoding="utf-8"))
            current_hash = self._compute_input_hash(step)
            return stored.get("input_hash") == current_hash
        except (json.JSONDecodeError, OSError):
            return False

    def _save_cache(self, step: PipelineStep) -> None:
        """Save cache metadata for a step."""
        cache_file = self.output_dir / f".cache_{step.name}.json"
        cache_data = {
            "input_hash": self._compute_input_hash(step),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "outputs": step.outputs,
        }
        cache_file.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")

    def _load_checkpoint(self) -> Dict[str, str]:
        """Load checkpoint of completed steps."""
        if not self.checkpoint_path.exists():
            return {}
        try:
            data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            return data.get("completed", {})
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_checkpoint(self, completed: Dict[str, str]) -> None:
        """Save checkpoint of completed steps."""
        data = {
            "completed": completed,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.checkpoint_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    def run(self, resume: bool = True) -> List[StepResult]:
        """Execute the pipeline in topological order.

        Args:
            resume: If True, skip steps that completed in a previous run.

        Returns:
            List of StepResult for each step.
        """
        order = self._topological_sort()
        completed = self._load_checkpoint() if resume else {}
        self._results = []

        logger.info(
            "Pipeline starting: %d steps [%s]",
            len(order),
            " → ".join(order),
        )

        for step_name in order:
            step = self._steps[step_name]

            # Check if already completed in a previous run
            if resume and step_name in completed:
                # Verify outputs still exist
                outputs_exist = all(
                    (self.output_dir / o).exists() for o in step.outputs
                )
                if outputs_exist:
                    result = StepResult(
                        name=step_name,
                        status="skipped",
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    )
                    self._results.append(result)
                    logger.info("[SKIP] %s (completed in previous run)", step_name)
                    continue

            # Check cache
            if self._is_cached(step):
                result = StepResult(
                    name=step_name,
                    status="cached",
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                )
                self._results.append(result)
                logger.info("[CACHED] %s", step_name)
                completed[step_name] = "cached"
                self._save_checkpoint(completed)
                continue

            # Execute step
            logger.info("[START] %s", step_name)
            t0 = time.time()
            try:
                step.fn(self.output_dir, **step.kwargs)
                elapsed = time.time() - t0
                result = StepResult(
                    name=step_name,
                    status="ok",
                    elapsed_seconds=round(elapsed, 1),
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    output_path=step.outputs[0] if step.outputs else None,
                )
                self._results.append(result)
                logger.info("[OK] %s (%.1fs)", step_name, elapsed)

                # Update cache and checkpoint
                self._save_cache(step)
                completed[step_name] = "ok"
                self._save_checkpoint(completed)

            except Exception as exc:
                elapsed = time.time() - t0
                result = StepResult(
                    name=step_name,
                    status="failed",
                    elapsed_seconds=round(elapsed, 1),
                    error=str(exc),
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                )
                self._results.append(result)
                logger.error("[FAILED] %s: %s (%.1fs)", step_name, exc, elapsed)

                # Save progress so we can resume later
                self._save_checkpoint(completed)
                self._save_log()
                raise PipelineError(step_name, str(exc)) from exc

        self._save_log()
        logger.info("Pipeline complete: %d steps", len(self._results))
        return self._results

    def _save_log(self) -> None:
        """Persist pipeline log to disk."""
        log_entries = [
            {
                "step": r.name,
                "status": r.status,
                "elapsed_seconds": r.elapsed_seconds,
                "error": r.error,
                "output_path": r.output_path,
                "timestamp": r.timestamp,
            }
            for r in self._results
        ]
        self.log_path.write_text(
            json.dumps(log_entries, indent=2), encoding="utf-8"
        )

    def get_results(self) -> List[StepResult]:
        """Return results from the most recent run."""
        return list(self._results)

    def clear_cache(self) -> None:
        """Remove all cache files."""
        for cache_file in self.output_dir.glob(".cache_*.json"):
            cache_file.unlink()
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        logger.info("Pipeline cache cleared")


class PipelineError(Exception):
    """Raised when a pipeline step fails."""

    def __init__(self, step_name: str, message: str) -> None:
        self.step_name = step_name
        super().__init__(f"Pipeline failed at step {step_name!r}: {message}")
