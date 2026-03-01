"""Tests for DAG pipeline runner."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from training_data_robo.pipeline import Pipeline, PipelineError, StepResult


def _step_write(output_dir: Path, filename: str = "out.txt", content: str = "ok") -> None:
    """Test step that writes a file."""
    (output_dir / filename).write_text(content, encoding="utf-8")


def _step_fail(output_dir: Path) -> None:
    """Test step that always fails."""
    raise RuntimeError("intentional failure")


def _step_append(output_dir: Path, filename: str = "log.txt", text: str = "x") -> None:
    """Test step that appends to a file."""
    path = output_dir / filename
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(existing + text, encoding="utf-8")


class TestTopologicalSort:
    def test_linear_chain(self, tmp_path: Path) -> None:
        pipe = Pipeline(tmp_path)
        pipe.add_step("a", _step_write)
        pipe.add_step("b", _step_write, depends_on=["a"])
        pipe.add_step("c", _step_write, depends_on=["b"])
        order = pipe._topological_sort()
        assert order == ["a", "b", "c"]

    def test_diamond_dependency(self, tmp_path: Path) -> None:
        pipe = Pipeline(tmp_path)
        pipe.add_step("a", _step_write)
        pipe.add_step("b", _step_write, depends_on=["a"])
        pipe.add_step("c", _step_write, depends_on=["a"])
        pipe.add_step("d", _step_write, depends_on=["b", "c"])
        order = pipe._topological_sort()
        assert order[0] == "a"
        assert order[-1] == "d"
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_independent_steps(self, tmp_path: Path) -> None:
        pipe = Pipeline(tmp_path)
        pipe.add_step("x", _step_write)
        pipe.add_step("y", _step_write)
        pipe.add_step("z", _step_write)
        order = pipe._topological_sort()
        assert set(order) == {"x", "y", "z"}

    def test_circular_dependency_raises(self, tmp_path: Path) -> None:
        pipe = Pipeline(tmp_path)
        pipe.add_step("a", _step_write)
        pipe.add_step("b", _step_write, depends_on=["a"])
        # Manually create a cycle by manipulating internals
        pipe._steps["a"].depends_on = ["b"]
        with pytest.raises(ValueError, match="circular"):
            pipe._topological_sort()


class TestPipelineRun:
    def test_simple_pipeline(self, tmp_path: Path) -> None:
        pipe = Pipeline(tmp_path, cache_enabled=False)
        pipe.add_step("write_a", _step_write, outputs=["out.txt"],
                       filename="out.txt", content="hello")
        results = pipe.run(resume=False)
        assert len(results) == 1
        assert results[0].status == "ok"
        assert (tmp_path / "out.txt").read_text() == "hello"

    def test_multi_step_pipeline(self, tmp_path: Path) -> None:
        pipe = Pipeline(tmp_path, cache_enabled=False)
        pipe.add_step("step1", _step_append, outputs=["log.txt"],
                       filename="log.txt", text="A")
        pipe.add_step("step2", _step_append, depends_on=["step1"],
                       outputs=["log.txt"], filename="log.txt", text="B")
        results = pipe.run(resume=False)
        assert len(results) == 2
        assert all(r.status == "ok" for r in results)
        assert (tmp_path / "log.txt").read_text() == "AB"

    def test_failure_stops_pipeline(self, tmp_path: Path) -> None:
        pipe = Pipeline(tmp_path, cache_enabled=False)
        pipe.add_step("step1", _step_write, outputs=["out.txt"],
                       filename="out.txt", content="ok")
        pipe.add_step("step2", _step_fail, depends_on=["step1"])
        with pytest.raises(PipelineError, match="step2"):
            pipe.run(resume=False)
        # step1 should have run
        assert (tmp_path / "out.txt").exists()

    def test_failure_saves_checkpoint(self, tmp_path: Path) -> None:
        pipe = Pipeline(tmp_path, cache_enabled=False)
        pipe.add_step("good", _step_write, outputs=["out.txt"],
                       filename="out.txt", content="ok")
        pipe.add_step("bad", _step_fail, depends_on=["good"])
        with pytest.raises(PipelineError):
            pipe.run(resume=False)

        # Checkpoint should have the successful step
        cp = json.loads(pipe.checkpoint_path.read_text())
        assert "good" in cp["completed"]
        assert "bad" not in cp["completed"]

    def test_pipeline_log_saved(self, tmp_path: Path) -> None:
        pipe = Pipeline(tmp_path, cache_enabled=False)
        pipe.add_step("step1", _step_write, outputs=["out.txt"],
                       filename="out.txt", content="ok")
        pipe.run(resume=False)
        log = json.loads(pipe.log_path.read_text())
        assert len(log) == 1
        assert log[0]["step"] == "step1"
        assert log[0]["status"] == "ok"

    def test_elapsed_time_recorded(self, tmp_path: Path) -> None:
        pipe = Pipeline(tmp_path, cache_enabled=False)
        pipe.add_step("step1", _step_write, outputs=["out.txt"],
                       filename="out.txt", content="ok")
        results = pipe.run(resume=False)
        assert results[0].elapsed_seconds >= 0


class TestCaching:
    def test_cache_skips_unchanged(self, tmp_path: Path) -> None:
        # First run
        pipe = Pipeline(tmp_path, cache_enabled=True)
        pipe.add_step("step1", _step_append, outputs=["log.txt"],
                       inputs=[], filename="log.txt", text="A")
        pipe.run(resume=False)
        assert (tmp_path / "log.txt").read_text() == "A"

        # Second run — should be cached
        pipe2 = Pipeline(tmp_path, cache_enabled=True)
        pipe2.add_step("step1", _step_append, outputs=["log.txt"],
                        inputs=[], filename="log.txt", text="A")
        results = pipe2.run(resume=False)
        assert results[0].status == "cached"
        # File should NOT have been appended to again
        assert (tmp_path / "log.txt").read_text() == "A"

    def test_cache_invalidated_on_input_change(self, tmp_path: Path) -> None:
        # Create initial input
        (tmp_path / "input.txt").write_text("v1", encoding="utf-8")

        pipe = Pipeline(tmp_path, cache_enabled=True)
        pipe.add_step("step1", _step_append, outputs=["log.txt"],
                       inputs=["input.txt"], filename="log.txt", text="A")
        pipe.run(resume=False)

        # Change input
        (tmp_path / "input.txt").write_text("v2", encoding="utf-8")

        pipe2 = Pipeline(tmp_path, cache_enabled=True)
        pipe2.add_step("step1", _step_append, outputs=["log.txt"],
                        inputs=["input.txt"], filename="log.txt", text="B")
        results = pipe2.run(resume=False)
        assert results[0].status == "ok"  # Re-executed due to input change

    def test_clear_cache(self, tmp_path: Path) -> None:
        pipe = Pipeline(tmp_path, cache_enabled=True)
        pipe.add_step("step1", _step_write, outputs=["out.txt"],
                       filename="out.txt", content="ok")
        pipe.run(resume=False)

        # Cache file should exist
        assert list(tmp_path.glob(".cache_*.json"))

        pipe.clear_cache()
        assert not list(tmp_path.glob(".cache_*.json"))


class TestResume:
    def test_resume_skips_completed(self, tmp_path: Path) -> None:
        # First run: step1 succeeds, step2 fails
        pipe = Pipeline(tmp_path, cache_enabled=False)
        pipe.add_step("step1", _step_append, outputs=["log.txt"],
                       filename="log.txt", text="A")
        pipe.add_step("step2", _step_fail, depends_on=["step1"])
        with pytest.raises(PipelineError):
            pipe.run(resume=False)

        # Resume: step1 should be skipped, step2 re-executed
        pipe2 = Pipeline(tmp_path, cache_enabled=False)
        pipe2.add_step("step1", _step_append, outputs=["log.txt"],
                        filename="log.txt", text="A")
        pipe2.add_step("step2", _step_write, depends_on=["step1"],
                        outputs=["out2.txt"], filename="out2.txt", content="ok")
        results = pipe2.run(resume=True)
        assert results[0].status == "skipped"
        assert results[1].status == "ok"


class TestEdgeCases:
    def test_duplicate_step_name_raises(self, tmp_path: Path) -> None:
        pipe = Pipeline(tmp_path)
        pipe.add_step("step1", _step_write)
        with pytest.raises(ValueError, match="already registered"):
            pipe.add_step("step1", _step_write)

    def test_missing_dependency_raises(self, tmp_path: Path) -> None:
        pipe = Pipeline(tmp_path)
        with pytest.raises(ValueError, match="hasn't been added"):
            pipe.add_step("step2", _step_write, depends_on=["nonexistent"])

    def test_empty_pipeline(self, tmp_path: Path) -> None:
        pipe = Pipeline(tmp_path)
        results = pipe.run()
        assert results == []

    def test_get_results(self, tmp_path: Path) -> None:
        pipe = Pipeline(tmp_path, cache_enabled=False)
        pipe.add_step("step1", _step_write, outputs=["out.txt"],
                       filename="out.txt", content="ok")
        pipe.run(resume=False)
        results = pipe.get_results()
        assert len(results) == 1
        assert isinstance(results[0], StepResult)
