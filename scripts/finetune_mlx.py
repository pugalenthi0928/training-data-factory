#!/usr/bin/env python3
"""LoRA fine-tune a small language model locally using Apple MLX.

Converts JSONL training data to MLX chat format, applies LoRA (rank 8),
trains for a configurable number of epochs, and saves adapters + metrics.

Usage:
    python scripts/finetune_mlx.py \
        --train-data output/train.jsonl \
        --test-data  output/test.jsonl \
        --output-dir runs/finetune_001 \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --epochs 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from training_data_robo.io import load_jsonl


def jsonl_to_chat_format(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert training data rows to MLX chat format.

    MLX expects: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    chat_rows = []
    for row in rows:
        input_text = str(row.get("input_text", ""))
        output_text = str(row.get("output_text", ""))
        if not input_text or not output_text:
            continue
        chat_rows.append(
            {
                "messages": [
                    {"role": "user", "content": input_text},
                    {"role": "assistant", "content": output_text},
                ]
            }
        )
    return chat_rows


def write_chat_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_finetune(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the MLX LoRA fine-tuning pipeline."""
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        print(
            "ERROR: mlx-lm not installed. Install with: pip install -e '.[mlx]'\n"
            "Note: MLX requires Apple Silicon (M1/M2/M3/M4).",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and convert data
    train_rows = load_jsonl(Path(args.train_data))
    test_rows = load_jsonl(Path(args.test_data)) if args.test_data else []

    if not train_rows:
        raise SystemExit("Training data is empty.")

    train_chat = jsonl_to_chat_format(train_rows)
    test_chat = jsonl_to_chat_format(test_rows) if test_rows else []

    print(f"Training examples: {len(train_chat)}")
    print(f"Test examples: {len(test_chat)}")

    # Write MLX-formatted data
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    write_chat_jsonl(data_dir / "train.jsonl", train_chat)
    if test_chat:
        write_chat_jsonl(data_dir / "valid.jsonl", test_chat)
        write_chat_jsonl(data_dir / "test.jsonl", test_chat)

    # Save config
    config = {
        "model": args.model,
        "train_examples": len(train_chat),
        "test_examples": len(test_chat),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lora_rank": args.lora_rank,
        "learning_rate": args.learning_rate,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Build LoRA training args
    num_iters = args.epochs * max(1, len(train_chat) // args.batch_size)
    lora_args = [
        "--model",
        args.model,
        "--data",
        str(data_dir),
        "--train",
        "--adapter-path",
        str(output_dir / "adapters"),
        "--iters",
        str(num_iters),
        "--batch-size",
        str(args.batch_size),
        "--num-layers",
        str(args.lora_layers),
        "--learning-rate",
        str(args.learning_rate),
        "--steps-per-report",
        "5",
        "--save-every",
        str(max(10, num_iters // 3)),
    ]

    if test_chat:
        lora_args.extend(["--val-batches", str(min(10, len(test_chat)))])

    print(f"\nStarting LoRA fine-tuning: {args.model}")
    print(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.learning_rate}")
    print(f"  LoRA rank: {args.lora_rank}, Layers: {args.lora_layers}")
    print(f"  Output: {output_dir}\n")

    t0 = time.time()

    # Use mlx_lm CLI interface via subprocess for clean isolation
    import subprocess

    cmd = [sys.executable, "-m", "mlx_lm.lora"] + lora_args
    print(f"  Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\nWARNING: Fine-tuning exited with code {result.returncode}", file=sys.stderr)

    elapsed = time.time() - t0

    # Save completion metrics
    config["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    config["training_time_seconds"] = round(elapsed, 1)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"\nFine-tuning complete in {elapsed:.1f}s")
    print(f"Adapters saved to: {output_dir / 'adapters'}")

    return config


def main() -> None:
    ap = argparse.ArgumentParser(description="LoRA fine-tune with MLX on Apple Silicon.")
    ap.add_argument("--train-data", required=True, help="Training JSONL")
    ap.add_argument("--test-data", default=None, help="Test/validation JSONL")
    ap.add_argument("--output-dir", required=True, help="Output directory for adapters and metrics")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace model ID")
    ap.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    ap.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    ap.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (default: 8)")
    ap.add_argument("--lora-layers", type=int, default=16, help="Number of LoRA layers (default: 16)")
    ap.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    args = ap.parse_args()

    run_finetune(args)


if __name__ == "__main__":
    main()
