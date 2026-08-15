"""Lightweight experiment tracker with no MLflow dependency.

Each run creates a directory under ``runs/{run_id}/`` containing:

- ``config.json``: hyperparameters and settings
- ``metrics.json``: accumulated metrics such as quality scores and training loss
- ``pipeline_log.json``: step-by-step execution log
- ``artifacts/``: saved files such as model adapters and reports

Supports comparing metrics across multiple runs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logging_config import get_logger

logger = get_logger("training_data_robo.tracker")


@dataclass
class RunInfo:
    """Metadata about a tracked run."""

    run_id: str
    run_dir: Path
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "running"  # running, completed, failed
    started_at: str = ""
    completed_at: Optional[str] = None


class ExperimentTracker:
    """Tracks experiment runs with configs, metrics, and artifacts.

    Usage::

        tracker = ExperimentTracker(runs_dir=Path("runs"))
        run = tracker.start_run("forge_001", config={"model": "gpt-4.1-mini"})
        tracker.log_metric("quality_mean", 4.2)
        tracker.log_metric("contamination_hits", 0)
        tracker.log_metrics({"rouge1": 0.45, "rouge2": 0.22})
        tracker.end_run()
    """

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._current_run: Optional[RunInfo] = None

    @property
    def current_run(self) -> Optional[RunInfo]:
        return self._current_run

    def start_run(
        self,
        run_id: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> RunInfo:
        """Start a new tracked experiment run."""
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts").mkdir(exist_ok=True)

        started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        run_config = config or {}
        run_config["run_id"] = run_id
        run_config["started_at"] = started_at

        run_info = RunInfo(
            run_id=run_id,
            run_dir=run_dir,
            config=run_config,
            status="running",
            started_at=started_at,
        )

        # Persist config
        (run_dir / "config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

        # Initialize empty metrics
        (run_dir / "metrics.json").write_text("{}\n", encoding="utf-8")

        self._current_run = run_info
        logger.info("Started run %s at %s", run_id, run_dir)
        return run_info

    def log_metric(self, key: str, value: Any) -> None:
        """Log a single metric for the current run."""
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        self._current_run.metrics[key] = value
        self._persist_metrics()

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log multiple metrics at once."""
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        self._current_run.metrics.update(metrics)
        self._persist_metrics()

    def log_artifact(self, name: str, data: Any) -> Path:
        """Save a JSON-serializable artifact."""
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        artifact_path = self._current_run.run_dir / "artifacts" / name
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(data, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Saved artifact: %s", artifact_path)
        return artifact_path

    def end_run(self, status: str = "completed") -> RunInfo:
        """Mark the current run as finished."""
        if self._current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        self._current_run.status = status
        self._current_run.completed_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        # Update config with completion info
        self._current_run.config["status"] = status
        self._current_run.config["completed_at"] = self._current_run.completed_at
        (self._current_run.run_dir / "config.json").write_text(
            json.dumps(self._current_run.config, indent=2, default=str),
            encoding="utf-8",
        )

        self._persist_metrics()
        logger.info("Ended run %s [%s]", self._current_run.run_id, status)

        run_info = self._current_run
        self._current_run = None
        return run_info

    def _persist_metrics(self) -> None:
        """Write current metrics to disk."""
        if self._current_run is None:
            return
        metrics_path = self._current_run.run_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(self._current_run.metrics, indent=2, default=str),
            encoding="utf-8",
        )

    def list_runs(self) -> List[RunInfo]:
        """List all tracked runs, sorted by start time (newest first)."""
        runs: List[RunInfo] = []
        for run_dir in sorted(self.runs_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            config_path = run_dir / "config.json"
            metrics_path = run_dir / "metrics.json"
            if not config_path.exists():
                continue

            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                metrics = {}
                if metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

                runs.append(
                    RunInfo(
                        run_id=config.get("run_id", run_dir.name),
                        run_dir=run_dir,
                        config=config,
                        metrics=metrics,
                        status=config.get("status", "unknown"),
                        started_at=config.get("started_at", ""),
                        completed_at=config.get("completed_at"),
                    )
                )
            except (json.JSONDecodeError, OSError):
                logger.warning("Skipping corrupt run dir: %s", run_dir)

        return runs

    def get_run(self, run_id: str) -> Optional[RunInfo]:
        """Load a specific run by ID."""
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            return None

        config_path = run_dir / "config.json"
        if not config_path.exists():
            return None

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            metrics = {}
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

            return RunInfo(
                run_id=run_id,
                run_dir=run_dir,
                config=config,
                metrics=metrics,
                status=config.get("status", "unknown"),
                started_at=config.get("started_at", ""),
                completed_at=config.get("completed_at"),
            )
        except (json.JSONDecodeError, OSError):
            return None

    def compare_runs(
        self,
        run_ids: List[str],
        metric_keys: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Compare metrics across multiple runs.

        Returns a list of dicts, each containing run_id and requested metrics.
        """
        rows: List[Dict[str, Any]] = []
        for run_id in run_ids:
            run = self.get_run(run_id)
            if run is None:
                logger.warning("Run %s not found, skipping", run_id)
                continue

            row: Dict[str, Any] = {"run_id": run_id, "status": run.status}
            if metric_keys:
                for key in metric_keys:
                    row[key] = run.metrics.get(key)
            else:
                row.update(run.metrics)
            rows.append(row)

        return rows
