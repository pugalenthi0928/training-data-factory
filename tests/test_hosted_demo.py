"""Hosted demonstration tests over the real Forge workflow."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from forge.hosted import HostedRateLimitError, HostedSettings, JobManager, RunRequest
from forge.web import create_app


def _settings(tmp_path: Path, **overrides: int) -> HostedSettings:
    values = {
        "data_dir": tmp_path / "hosted",
        "max_workers": 1,
        "max_jobs": 8,
        "ttl_seconds": 600,
        "rate_limit": 8,
        "rate_window_seconds": 600,
        "min_document_chars": 320,
        "max_document_chars": 12_000,
        "max_total_chars": 24_000,
    }
    values.update(overrides)
    return HostedSettings(**values)


def test_hosted_preset_runs_real_pipeline_and_serves_verified_evidence(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        response = client.post("/api/runs", json={"preset": "release-controls"})
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        status = app.state.job_manager.wait(job_id, timeout=30)

        assert status["status"] == "succeeded"
        assert status["completed_stages"] == 12
        assert all(stage["status"] == "passed" for stage in status["stages"])
        assert status["summary"]["verified"] is True
        assert status["summary"]["records"] > 0
        assert status["summary"]["source_overlap"] == 0
        assert status["summary"]["contamination_flags"] == 0
        assert status["summary"]["claim_status"]["model_quality"] == "not_established_by_this_release"

        artifact_response = client.get(f"/api/runs/{job_id}/artifacts")
        assert artifact_response.status_code == 200
        artifacts = artifact_response.json()["artifacts"]
        assert {artifact["key"] for artifact in artifacts} >= {
            "release",
            "events",
            "source-governance",
            "contamination",
            "split",
            "train",
            "test",
        }

        release_response = client.get(f"/api/runs/{job_id}/artifacts/release")
        assert release_response.status_code == 200
        assert release_response.json()["release_id"] == status["summary"]["release_id"]

        bundle_response = client.get(f"/api/runs/{job_id}/download")
        assert bundle_response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(bundle_response.content)) as bundle:
            names = set(bundle.namelist())
        assert "README.txt" in names
        assert "release_manifest.json" in names
        assert "train.jsonl" in names
        assert not any(name.startswith(".forge/") or name.startswith("inputs/") for name in names)


def test_duplicate_submission_reuses_content_addressed_job(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        first = client.post("/api/runs", json={"preset": "incident-operations"})
        second = client.post("/api/runs", json={"preset": "incident-operations"})

        assert first.status_code == 202
        assert second.status_code == 200
        assert second.json()["reused"] is True
        assert second.json()["job_id"] == first.json()["job_id"]
        app.state.job_manager.wait(first.json()["job_id"], timeout=30)


def test_custom_input_contract_rejects_short_or_ambiguous_sources(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        short = client.post(
            "/api/runs",
            json={
                "preset": None,
                "documents": [
                    {"title": "One", "text": "too short"},
                    {"title": "Two", "text": "also too short"},
                ],
            },
        )
        assert short.status_code == 422
        assert "at least 320 characters" in short.json()["detail"]

        both = client.post(
            "/api/runs",
            json={
                "preset": "release-controls",
                "documents": [
                    {"title": "One", "text": "alpha " * 80},
                    {"title": "Two", "text": "beta " * 80},
                ],
            },
        )
        assert both.status_code == 422
        assert "not both" in both.json()["detail"]


def test_security_headers_keep_the_product_strict_and_api_docs_functional(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        product = client.get("/")
        docs = client.get("/api/docs")

        assert product.status_code == 200
        assert "script-src 'self'; style-src 'self'" in product.headers["content-security-policy"]
        assert "unsafe-inline" not in product.headers["content-security-policy"]
        assert product.headers["x-frame-options"] == "DENY"
        assert docs.status_code == 200
        assert "https://cdn.jsdelivr.net" in docs.headers["content-security-policy"]


def test_artifact_route_is_allowlisted(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    with TestClient(app) as client:
        response = client.post("/api/runs", json={"preset": "regulated-change"})
        job_id = response.json()["job_id"]
        app.state.job_manager.wait(job_id, timeout=30)

        assert client.get(f"/api/runs/{job_id}/artifacts/runtime-inputs").status_code == 404
        assert client.get(f"/api/runs/{job_id}/artifacts/.forge").status_code == 404


def test_rate_limit_applies_to_new_runs_but_not_duplicate_requests(tmp_path: Path) -> None:
    manager = JobManager(_settings(tmp_path, rate_limit=1))
    try:
        first, reused = manager.submit(RunRequest(preset="release-controls"), client_key="client")
        assert reused is False
        duplicate, reused = manager.submit(RunRequest(preset="release-controls"), client_key="client")
        assert reused is True
        assert duplicate.job_id == first.job_id

        try:
            manager.submit(RunRequest(preset="incident-operations"), client_key="client")
        except HostedRateLimitError:
            pass
        else:
            raise AssertionError("A distinct second run should be rate limited")
        manager.wait(first.job_id, timeout=30)
    finally:
        manager.close()


def test_public_copy_states_claim_boundary_and_avoids_em_dashes() -> None:
    static_dir = Path(__file__).parents[1] / "src" / "forge" / "web_static"
    copy = "\n".join(path.read_text(encoding="utf-8") for path in static_dir.iterdir() if path.is_file())

    assert "SCOPE OF THIS DEMONSTRATION" in copy
    assert "does not evaluate model quality or establish production safety" in copy
    assert "—" not in copy
