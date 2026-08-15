"""End-to-end evidence tests for the shared Forge workflow API."""

from __future__ import annotations

import json
from pathlib import Path

from forge import ForgeConfig, run_forge
from training_data_robo.releases import verify_release


def _fixture_config(tmp_path: Path) -> ForgeConfig:
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "one.txt").write_text(
        "Alpha systems preserve source lineage through every transformation. " * 8,
        encoding="utf-8",
    )
    (sources / "two.txt").write_text(
        "Beta systems verify output artifacts before a dataset is released. " * 8,
        encoding="utf-8",
    )
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        json.dumps({"question": "How do independent weather sensors measure rainfall in remote valleys?"}) + "\n",
        encoding="utf-8",
    )
    return ForgeConfig.from_paths(
        sources=[str(sources)],
        benchmarks=[str(benchmark)],
        tasks=("qa", "summary"),
        max_examples=20,
        max_chars=280,
        dry_run=True,
    )


def test_real_workflow_resumes_with_same_release_identity(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    run_dir = tmp_path / "run"
    first = run_forge(run_dir, config)
    second = run_forge(run_dir, config)

    assert first.release_id == second.release_id
    assert first.cache_hits == 0
    assert second.cache_hits == len(second.stage_results) == 9
    assert verify_release(run_dir / "release_manifest.json")["verified"] is True
    assert (run_dir / "pipeline_events.jsonl").is_file()
    assert (run_dir / ".forge" / "state.json").is_file()
    pipeline_log = json.loads((run_dir / "pipeline_log.json").read_text(encoding="utf-8"))
    assert all(entry["status"] == "ok" for entry in pipeline_log)
    assert all(entry["execution"] == "cached" for entry in pipeline_log)


def test_workflow_repairs_tampered_intermediate_artifact(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    run_dir = tmp_path / "run"
    initial = run_forge(run_dir, config)
    (run_dir / "quality.jsonl").write_text("tampered\n", encoding="utf-8")

    repaired = run_forge(run_dir, config)

    statuses = {result.name: result.status for result in repaired.stage_results}
    assert statuses["ingest"] == "cached"
    assert statuses["generate"] == "cached"
    assert statuses["quality"] == "ok"
    assert repaired.release_id == initial.release_id
    assert verify_release(run_dir / "release_manifest.json")["verified"] is True
