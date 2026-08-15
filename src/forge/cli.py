"""Command-line entry point for the canonical Forge workflow API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import StageExecutionError
from .workflow import ForgeConfig, run_forge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forge: verifiable training-data curation pipeline")
    parser.add_argument("--source", required=True, action="append", help="Source folder or file; repeatable")
    parser.add_argument("--source-manifest", default=None, help="JSON or JSONL source-rights policy manifest")
    parser.add_argument("--output-dir", required=True, help="Run output directory")
    parser.add_argument("--tasks", default="qa,summary", help="Comma-separated task types")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Generation model")
    parser.add_argument("--judge-model", default="gpt-4.1-mini", help="Judge model")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--select-n", type=int, default=None)
    parser.add_argument("--select-strategy", default="quality_weighted")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--pii-action", choices=("reject", "redact"), default="reject")
    parser.add_argument("--fuzzy-dedupe-threshold", type=float, default=0.8)
    parser.add_argument("--fuzzy-contamination-threshold", type=float, default=0.8)
    parser.add_argument(
        "--semantic-backend",
        choices=("disabled", "sentence_transformers", "openai"),
        default="disabled",
    )
    parser.add_argument("--semantic-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--semantic-revision", default="main")
    parser.add_argument("--semantic-dedupe-threshold", type=float, default=0.9)
    parser.add_argument("--semantic-contamination-threshold", type=float, default=0.9)
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic local generator and judge")
    parser.add_argument("--skip-finetune", action="store_true", help="Retained for compatibility; curation only")
    parser.add_argument("--ft-model", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ft-epochs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--benchmark-file",
        action="append",
        required=True,
        help="Independent benchmark JSONL; repeatable",
    )
    parser.add_argument("--dataset-name", default="forge-generated-dataset")
    parser.add_argument("--dataset-version", default="0.1.0")
    parser.add_argument("--dataset-license", default=None)
    parser.add_argument("--benchmark-origin", default=None)
    parser.add_argument("--release-tier", choices=("smoke", "candidate"), default=None)
    parser.add_argument("--no-resume", action="store_true", help="Execute every stage even when valid cache exists")
    parser.add_argument("--no-cache", action="store_true", help="Disable stage cache reads for this run")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run:
        dataset_license = args.dataset_license or "NOASSERTION"
        benchmark_origin = args.benchmark_origin or "repository synthetic contamination fixture"
        release_tier = args.release_tier or "smoke"
    else:
        missing = [
            flag
            for flag, value in (
                ("--dataset-license", args.dataset_license),
                ("--benchmark-origin", args.benchmark_origin),
                ("--source-manifest", args.source_manifest),
            )
            if not value
        ]
        if missing:
            parser.error(f"live runs require release metadata: {', '.join(missing)}")
        if not args.skip_finetune:
            parser.error(
                "Forge keeps curation and release in one pipeline. Portable training is a later stage; "
                "pass --skip-finetune for this release workflow."
            )
        if args.semantic_backend == "disabled":
            parser.error("candidate runs require --semantic-backend sentence_transformers or openai")
        dataset_license = args.dataset_license
        benchmark_origin = args.benchmark_origin
        release_tier = args.release_tier or "candidate"

    config = ForgeConfig.from_paths(
        sources=args.source,
        benchmarks=args.benchmark_file,
        source_manifest=args.source_manifest,
        tasks=tuple(task.strip() for task in args.tasks.split(",") if task.strip()),
        model=args.model,
        judge_model=args.judge_model,
        max_examples=args.max_examples,
        max_chars=args.max_chars,
        overlap=args.overlap,
        select_n=args.select_n,
        select_strategy=args.select_strategy,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        pii_action=args.pii_action,
        fuzzy_dedupe_threshold=args.fuzzy_dedupe_threshold,
        fuzzy_contamination_threshold=args.fuzzy_contamination_threshold,
        semantic_backend=args.semantic_backend,
        semantic_model=args.semantic_model,
        semantic_revision=args.semantic_revision,
        semantic_dedupe_threshold=args.semantic_dedupe_threshold,
        semantic_contamination_threshold=args.semantic_contamination_threshold,
        dry_run=args.dry_run,
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        dataset_license=dataset_license,
        benchmark_origin=benchmark_origin,
        release_tier=release_tier,
    )
    try:
        result = run_forge(
            Path(args.output_dir),
            config,
            resume=not args.no_resume,
            cache_enabled=not args.no_cache,
        )
    except StageExecutionError as exc:
        parser.exit(1, f"{exc}\nResume with the same command after correcting the failure.\n")

    print(
        json.dumps(
            {
                "status": "complete",
                "run_dir": str(result.run_dir),
                "release_id": result.release_id,
                "release_tier": result.release_tier,
                "stages": len(result.stage_results),
                "cache_hits": result.cache_hits,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
