"""FastAPI entry point for the bounded Forge hosted demonstration."""

from __future__ import annotations

import hashlib
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .hosted import (
    HostedRateLimitError,
    HostedSettings,
    HostedValidationError,
    JobManager,
    RunRequest,
)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def settings_from_environment() -> HostedSettings:
    return HostedSettings(
        data_dir=Path(os.getenv("FORGE_DATA_DIR", "/tmp/forge-hosted")),
        max_workers=_env_int("FORGE_MAX_WORKERS", 1),
        max_jobs=_env_int("FORGE_MAX_JOBS", 24),
        ttl_seconds=_env_int("FORGE_JOB_TTL_SECONDS", 3_600),
        rate_limit=_env_int("FORGE_RATE_LIMIT", 12),
        rate_window_seconds=_env_int("FORGE_RATE_WINDOW_SECONDS", 3_600),
    )


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "").split(",", maxsplit=1)[0].strip()
    host = forwarded or (request.client.host if request.client else "unknown")
    user_agent = request.headers.get("user-agent", "unknown")[:200]
    return hashlib.sha256(f"{host}|{user_agent}".encode()).hexdigest()


def create_app(settings: HostedSettings | None = None) -> FastAPI:
    manager = JobManager(settings or settings_from_environment())
    static_dir = Path(__file__).with_name("web_static")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        manager.close()

    app = FastAPI(
        title="Forge hosted demonstration",
        version="0.13.0",
        description=(
            "A bounded interface to the real Forge Python pipeline. Public runs are deterministic smoke releases."
        ),
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.job_manager = manager

    @app.middleware("http")
    async def security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        if request.url.path.startswith("/api/docs"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data:; "
                "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
            )
        else:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
            )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        if request.url.path.startswith("/api/runs"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html", media_type="text/html")

    @app.get("/health", tags=["operations"])
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "forge-hosted", "mode": "deterministic-smoke"}

    @app.get("/api/meta", tags=["demonstration"])
    def meta() -> dict[str, object]:
        return {
            "pipeline": "forge.workflow.run_forge",
            "stages": 12,
            "mode": "deterministic-smoke",
            "claim": {
                "established": [
                    "pipeline execution",
                    "artifact integrity",
                    "source-isolated split",
                    "declared lexical and fuzzy controls",
                ],
                "not_established": ["model quality", "production safety", "human preference"],
            },
        }

    @app.get("/api/presets", tags=["demonstration"])
    def presets() -> dict[str, object]:
        return {"presets": manager.presets(), "custom_document_count": 2}

    @app.post("/api/runs", tags=["runs"])
    def create_run(payload: RunRequest, request: Request) -> JSONResponse:
        try:
            record, reused = manager.submit(payload, client_key=_client_key(request))
        except HostedValidationError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
        except HostedRateLimitError as exc:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
        body = {
            "job_id": record.job_id,
            "status": record.status,
            "reused": reused,
            "status_url": f"/api/runs/{record.job_id}",
        }
        return JSONResponse(body, status_code=200 if reused else status.HTTP_202_ACCEPTED)

    @app.get("/api/runs/{job_id}", tags=["runs"])
    def run_status(job_id: str) -> dict[str, object]:
        try:
            return manager.status(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found or expired.") from exc

    @app.get("/api/runs/{job_id}/artifacts", tags=["artifacts"])
    def artifacts(job_id: str) -> dict[str, object]:
        try:
            return {"artifacts": manager.artifact_list(job_id)}
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found or expired.") from exc
        except HostedValidationError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    @app.get("/api/runs/{job_id}/artifacts/{artifact_key}", tags=["artifacts"])
    def artifact(job_id: str, artifact_key: str) -> FileResponse:
        try:
            path, media_type = manager.artifact(job_id, artifact_key)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found.") from exc
        except HostedValidationError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        return FileResponse(path, media_type=media_type, filename=path.name, content_disposition_type="inline")

    @app.get("/api/runs/{job_id}/download", tags=["artifacts"])
    def download(job_id: str) -> Response:
        try:
            bundle = manager.evidence_bundle(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found or expired.") from exc
        except HostedValidationError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        return Response(
            bundle,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="forge-{job_id}-evidence.zip"'},
        )

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


app = create_app()


def main() -> None:
    import uvicorn

    port = _env_int("PORT", 8000)
    uvicorn.run("forge.web:app", host="0.0.0.0", port=port, proxy_headers=True)


if __name__ == "__main__":
    main()
