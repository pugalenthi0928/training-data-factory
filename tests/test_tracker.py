"""Tests for experiment tracker."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training_data_robo.tracker import ExperimentTracker


class TestStartRun:
    def test_creates_run_directory(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        run = tracker.start_run("run_001", config={"model": "gpt-4.1-mini"})
        assert run.run_dir.exists()
        assert (run.run_dir / "config.json").exists()
        assert (run.run_dir / "metrics.json").exists()
        assert (run.run_dir / "artifacts").is_dir()

    def test_config_persisted(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.start_run("run_001", config={"model": "test", "epochs": 3})
        config = json.loads((tmp_path / "run_001" / "config.json").read_text())
        assert config["model"] == "test"
        assert config["epochs"] == 3
        assert config["run_id"] == "run_001"
        assert "started_at" in config

    def test_default_config(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        run = tracker.start_run("run_002")
        assert run.config["run_id"] == "run_002"

    def test_current_run_set(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        assert tracker.current_run is None
        tracker.start_run("run_001")
        assert tracker.current_run is not None
        assert tracker.current_run.run_id == "run_001"


class TestLogMetrics:
    def test_log_single_metric(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.start_run("run_001")
        tracker.log_metric("quality_mean", 4.2)

        metrics = json.loads((tmp_path / "run_001" / "metrics.json").read_text())
        assert metrics["quality_mean"] == 4.2

    def test_log_multiple_metrics(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.start_run("run_001")
        tracker.log_metrics({"rouge1": 0.45, "rouge2": 0.22, "exact_match": 0.1})

        metrics = json.loads((tmp_path / "run_001" / "metrics.json").read_text())
        assert metrics["rouge1"] == 0.45
        assert metrics["rouge2"] == 0.22

    def test_metrics_accumulate(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.start_run("run_001")
        tracker.log_metric("step1_loss", 2.5)
        tracker.log_metric("step2_loss", 1.8)

        metrics = json.loads((tmp_path / "run_001" / "metrics.json").read_text())
        assert "step1_loss" in metrics
        assert "step2_loss" in metrics

    def test_no_active_run_raises(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_metric("x", 1)

    def test_log_metrics_no_run_raises(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_metrics({"x": 1})


class TestLogArtifact:
    def test_saves_json_artifact(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.start_run("run_001")
        path = tracker.log_artifact("report.json", {"accuracy": 0.95})
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["accuracy"] == 0.95

    def test_no_active_run_raises(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.log_artifact("x.json", {})


class TestEndRun:
    def test_marks_completed(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.start_run("run_001")
        run = tracker.end_run()
        assert run.status == "completed"
        assert run.completed_at is not None
        assert tracker.current_run is None

    def test_marks_failed(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.start_run("run_001")
        run = tracker.end_run(status="failed")
        assert run.status == "failed"

    def test_config_updated_on_end(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.start_run("run_001")
        tracker.end_run()
        config = json.loads((tmp_path / "run_001" / "config.json").read_text())
        assert config["status"] == "completed"
        assert "completed_at" in config

    def test_no_active_run_raises(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        with pytest.raises(RuntimeError, match="No active run"):
            tracker.end_run()


class TestListRuns:
    def test_lists_all_runs(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.start_run("run_001")
        tracker.end_run()
        tracker.start_run("run_002")
        tracker.end_run()

        runs = tracker.list_runs()
        run_ids = [r.run_id for r in runs]
        assert "run_001" in run_ids
        assert "run_002" in run_ids

    def test_empty_runs_dir(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        runs = tracker.list_runs()
        assert runs == []

    def test_skips_non_run_dirs(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        # Create a file (not a directory)
        (tmp_path / "readme.txt").write_text("hi")
        # Create a dir without config.json
        (tmp_path / "junk").mkdir()
        runs = tracker.list_runs()
        assert runs == []


class TestGetRun:
    def test_loads_existing_run(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.start_run("run_001", config={"model": "test"})
        tracker.log_metric("score", 4.5)
        tracker.end_run()

        run = tracker.get_run("run_001")
        assert run is not None
        assert run.run_id == "run_001"
        assert run.config["model"] == "test"
        assert run.metrics["score"] == 4.5

    def test_returns_none_for_missing(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        assert tracker.get_run("nonexistent") is None


class TestCompareRuns:
    def test_compare_two_runs(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.start_run("run_001")
        tracker.log_metrics({"rouge1": 0.40, "rouge2": 0.20})
        tracker.end_run()

        tracker.start_run("run_002")
        tracker.log_metrics({"rouge1": 0.50, "rouge2": 0.30})
        tracker.end_run()

        comparison = tracker.compare_runs(
            ["run_001", "run_002"],
            metric_keys=["rouge1", "rouge2"],
        )
        assert len(comparison) == 2
        assert comparison[0]["run_id"] == "run_001"
        assert comparison[0]["rouge1"] == 0.40
        assert comparison[1]["run_id"] == "run_002"
        assert comparison[1]["rouge1"] == 0.50

    def test_compare_all_metrics(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.start_run("run_001")
        tracker.log_metrics({"a": 1, "b": 2})
        tracker.end_run()

        comparison = tracker.compare_runs(["run_001"])
        assert comparison[0]["a"] == 1
        assert comparison[0]["b"] == 2

    def test_missing_run_skipped(self, tmp_path: Path) -> None:
        tracker = ExperimentTracker(tmp_path)
        tracker.start_run("run_001")
        tracker.end_run()

        comparison = tracker.compare_runs(["run_001", "nonexistent"])
        assert len(comparison) == 1
