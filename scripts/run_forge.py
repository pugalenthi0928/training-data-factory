#!/usr/bin/env python3
"""Forge: one-command pipeline orchestrating the full training data lifecycle.

    generate → quality → judge → contamination → difficulty →
    select → split → finetune → benchmark

Usage:
    python scripts/run_forge.py \
        --source ./sample_docs \
        --output-dir runs/forge_001 \
        --tasks qa,summary,instruction,cot \
        --model gpt-4.1-mini \
        --max-examples 200

    # Dry run (DummyLLM, no API calls):
    python scripts/run_forge.py \
        --source ./sample_docs \
        --output-dir runs/forge_dry \
        --tasks qa,summary \
        --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _run_step(name: str, cmd: List[str], output_dir: Path) -> bool:
    """Run a subprocess step with logging."""
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"\n  [{status}] {name} ({elapsed:.1f}s)")

    # Log step result
    log_path = output_dir / "pipeline_log.json"
    log_entries: List[Dict[str, Any]] = []
    if log_path.exists():
        import contextlib
        with contextlib.suppress(Exception):
            log_entries = json.loads(log_path.read_text(encoding="utf-8"))
    log_entries.append({
        "step": name,
        "status": status.lower(),
        "elapsed_seconds": round(elapsed, 1),
        "command": " ".join(cmd),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    log_path.write_text(json.dumps(log_entries, indent=2), encoding="utf-8")

    return result.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Forge: full training data pipeline.")
    ap.add_argument("--source", required=True, action="append", help="Source folder(s) for documents")
    ap.add_argument("--output-dir", required=True, help="Run output directory")
    ap.add_argument("--tasks", default="qa,summary", help="Comma-separated task types")
    ap.add_argument("--model", default="gpt-4.1-mini", help="LLM model for generation")
    ap.add_argument("--max-examples", type=int, default=200, help="Max examples to generate")
    ap.add_argument("--max-chars", type=int, default=900, help="Max chunk size")
    ap.add_argument("--select-n", type=int, default=None, help="Select top N for fine-tuning (default: all)")
    ap.add_argument("--select-strategy", default="quality_weighted", help="Selection strategy")
    ap.add_argument("--ft-model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model to fine-tune")
    ap.add_argument("--ft-epochs", type=int, default=3, help="Fine-tuning epochs")
    ap.add_argument("--dry-run", action="store_true", help="Use DummyLLM, skip fine-tuning/benchmark")
    ap.add_argument("--skip-finetune", action="store_true", help="Skip fine-tuning and benchmark steps")
    ap.add_argument("--benchmark-file", default=None, help="Benchmark JSONL for contamination check")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args).copy()
    config["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    py = sys.executable
    scripts = Path(__file__).resolve().parent

    # File paths
    raw = out / "raw_dataset.jsonl"
    quality = out / "quality.jsonl"
    deduped = out / "deduped.jsonl"
    judged = out / "judged.jsonl"
    difficulty_out = out / "difficulty.jsonl"
    selected = out / "selected.jsonl"
    train = out / "train.jsonl"
    test = out / "test.jsonl"

    # ---- Step 1: Generate ----
    source_flags = []
    for s in args.source:
        source_flags.extend(["-s", s])

    gen_cmd = [
        py, "-m", "training_data_robo.cli", "process",
        *source_flags,
        "-o", str(raw),
        "-t", args.tasks,
        "--max-examples", str(args.max_examples),
        "--max-chars", str(args.max_chars),
    ]
    if not args.dry_run:
        gen_cmd.extend(["--model", args.model])

    if not _run_step("Generate training data", gen_cmd, out):
        print("Pipeline failed at generation step.", file=sys.stderr)
        return

    # ---- Step 2: Quality scoring ----
    if not _run_step("Quality scoring", [
        py, str(scripts / "postprocess_quality.py"),
        "--input", str(raw), "--output", str(quality),
    ], out):
        quality = raw  # fallback to raw if quality step fails

    # ---- Step 3: Dedup ----
    if not _run_step("Deduplication", [
        py, str(scripts / "compute_dedupe.py"),
        "--input", str(quality), "--output", str(deduped),
        "--method", "hash", "--text-field", "output_text",
    ], out):
        deduped = quality

    # ---- Step 4: LLM-as-Judge ----
    judge_cmd = [
        py, str(scripts / "run_judge.py"),
        "--input", str(deduped), "--output", str(judged),
    ]
    if args.dry_run:
        judge_cmd.append("--fake")
    if not _run_step("LLM-as-Judge scoring", judge_cmd, out):
        judged = deduped

    # ---- Step 5: Contamination check ----
    if args.benchmark_file:
        _run_step("Contamination check", [
            py, str(scripts / "check_contamination.py"),
            "--input", str(judged),
            "--benchmark", args.benchmark_file,
            "--output", str(out / "contamination_report.json"),
        ], out)

    # ---- Step 6: Difficulty calibration ----
    if not _run_step("Difficulty calibration", [
        py, str(scripts / "calibrate_difficulty.py"),
        "--input", str(judged), "--output", str(difficulty_out),
    ], out):
        difficulty_out = judged

    # ---- Step 7: Selection (optional) ----
    final_dataset = difficulty_out
    if args.select_n:
        # Use Python inline for selection since it's a library call
        select_cmd = [
            py, "-c",
            f"""
import json, sys
sys.path.insert(0, 'src')
from training_data_robo.io import load_jsonl, write_jsonl
from training_data_robo.selector import select_examples
from pathlib import Path
rows = load_jsonl(Path('{difficulty_out}'))
selected = select_examples(rows, {args.select_n}, strategy='{args.select_strategy}')
write_jsonl(Path('{selected}'), selected)
print(json.dumps({{"selected": len(selected), "from": len(rows), "strategy": "{args.select_strategy}"}}))
""",
        ]
        if _run_step("Select examples", select_cmd, out):
            final_dataset = selected

    # ---- Step 8: Train/test split ----
    split_input = final_dataset
    if not _run_step("Train/test split", [
        py, str(scripts / "split_dataset.py"),
        "--input", str(split_input),
        "--train-output", str(train),
        "--test-output", str(test),
        "--test-fraction", "0.2",
    ], out):
        print("Pipeline failed at split step.", file=sys.stderr)
        return

    # ---- Step 9: Fine-tune (MLX) ----
    if not args.dry_run and not args.skip_finetune:
        ft_dir = out / "finetune"
        if _run_step("MLX LoRA fine-tuning", [
            py, str(scripts / "finetune_mlx.py"),
            "--train-data", str(train),
            "--test-data", str(test),
            "--output-dir", str(ft_dir),
            "--model", args.ft_model,
            "--epochs", str(args.ft_epochs),
        ], out):
            # ---- Step 10: Benchmark ----
            _run_step("Benchmark comparison", [
                py, str(scripts / "benchmark.py"),
                "--test-data", str(test),
                "--base-model", args.ft_model,
                "--finetuned-adapter", str(ft_dir / "adapters"),
                "--backend", "mlx",
                "--output", str(out / "benchmark_results.json"),
            ], out)
    elif args.dry_run:
        print("\n[DRY RUN] Skipping fine-tuning and benchmark steps.")
    else:
        print("\n[SKIP] Fine-tuning and benchmark steps skipped (--skip-finetune).")

    # ---- Done ----
    config["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    print(f"\n{'='*60}")
    print("  FORGE PIPELINE COMPLETE")
    print(f"  Output: {out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
