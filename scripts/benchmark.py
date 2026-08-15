#!/usr/bin/env python3
"""Benchmarking harness: compare base vs fine-tuned model on held-out test set.

Runs both models on the same test set, computes metrics (ROUGE, exact match),
and produces a comparison report with deltas.

Supports:
  - MLX local models (base + LoRA adapters)
  - OpenAI API models
  - Paired randomization testing
  - Paired bootstrap confidence intervals
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from training_data_robo.io import load_jsonl


def _compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores. Requires rouge-score package."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return {"rouge1_f": 0.0, "rouge2_f": 0.0, "rougeL_f": 0.0, "error": "rouge-score not installed"}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)

    n = max(1, len(predictions))
    return {
        "rouge1_f": round(sum(r1) / n, 4),
        "rouge2_f": round(sum(r2) / n, 4),
        "rougeL_f": round(sum(rl) / n, 4),
    }


def _compute_exact_match(predictions: List[str], references: List[str]) -> float:
    if not predictions:
        return 0.0
    matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return round(matches / len(predictions), 4)


def _paired_randomization_test(
    scores_a: List[float],
    scores_b: List[float],
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """One-sided paired randomization test for whether B exceeds A."""
    if len(scores_a) != len(scores_b):
        raise ValueError("Paired score lists must have equal length")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    rng = random.Random(seed)
    n = len(scores_a)
    if n == 0:
        return {
            "method": "paired_randomization_one_sided",
            "p_value": 1.0,
            "significant": False,
            "n_resamples": n_resamples,
            "seed": seed,
        }

    diffs = [b - a for a, b in zip(scores_a, scores_b)]
    observed_delta = sum(diffs) / n

    # Null hypothesis: no difference. Randomly flip signs of diffs.
    count_as_extreme = 0
    for _ in range(n_resamples):
        shuffled = sum(d * rng.choice([-1, 1]) for d in diffs) / n
        if shuffled >= observed_delta:
            count_as_extreme += 1

    # Plus-one correction prevents reporting an impossible p-value of exactly 0.
    p_value = (count_as_extreme + 1) / (n_resamples + 1)
    return {
        "method": "paired_randomization_one_sided",
        "observed_delta": round(observed_delta, 4),
        "p_value": p_value,
        "significant": p_value < 0.05,
        "n_resamples": n_resamples,
        "seed": seed,
    }


def _paired_bootstrap_ci(
    scores_a: List[float],
    scores_b: List[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Percentile bootstrap confidence interval for the paired mean delta."""
    if len(scores_a) != len(scores_b):
        raise ValueError("Paired score lists must have equal length")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if not scores_a:
        return {
            "method": "paired_percentile_bootstrap",
            "confidence": confidence,
            "lower": None,
            "upper": None,
            "n_resamples": n_resamples,
            "seed": seed,
        }

    rng = random.Random(seed)
    diffs = [b - a for a, b in zip(scores_a, scores_b)]
    n = len(diffs)
    sampled_means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))
    alpha = 1.0 - confidence
    lower_index = max(0, int((alpha / 2.0) * n_resamples))
    upper_index = min(
        n_resamples - 1,
        int((1.0 - alpha / 2.0) * n_resamples) - 1,
    )
    return {
        "method": "paired_percentile_bootstrap",
        "confidence": confidence,
        "lower": round(sampled_means[lower_index], 4),
        "upper": round(sampled_means[upper_index], 4),
        "n_resamples": n_resamples,
        "seed": seed,
    }


def generate_predictions_mlx(
    test_rows: List[Dict[str, Any]],
    model: str,
    adapter_path: Optional[str] = None,
    max_tokens: int = 256,
) -> List[str]:
    """Generate predictions using MLX local model."""
    try:
        from mlx_lm import generate, load
    except ImportError:
        print("ERROR: mlx-lm not installed. Install with: pip install -e '.[mlx]'", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {model}" + (f" + adapters: {adapter_path}" if adapter_path else ""))
    model_obj, tokenizer = load(model, adapter_path=adapter_path)

    predictions = []
    for i, row in enumerate(test_rows):
        input_text = str(row.get("input_text", ""))
        messages = [{"role": "user", "content": input_text}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        response = generate(model_obj, tokenizer, prompt=prompt, max_tokens=max_tokens)
        predictions.append(response)
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(test_rows)}")

    return predictions


def generate_predictions_openai(
    test_rows: List[Dict[str, Any]],
    model: str,
    max_tokens: int = 256,
) -> List[str]:
    """Generate predictions using OpenAI API."""
    from openai import OpenAI

    client = OpenAI()

    predictions = []
    for i, row in enumerate(test_rows):
        input_text = str(row.get("input_text", ""))
        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": input_text}],
            max_output_tokens=max_tokens,
        )
        predictions.append(response.output_text)
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(test_rows)}")

    return predictions


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    test_rows = load_jsonl(Path(args.test_data))
    if not test_rows:
        raise SystemExit("Test data is empty.")

    if args.max_examples:
        test_rows = test_rows[: args.max_examples]

    references = [str(r.get("output_text", "")) for r in test_rows]

    results: Dict[str, Any] = {
        "test_examples": len(test_rows),
        "test_data": args.test_data,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # --- Base model ---
    print(f"\n=== Base model: {args.base_model} ===")
    t0 = time.time()
    if args.backend == "mlx":
        base_preds = generate_predictions_mlx(test_rows, args.base_model, max_tokens=args.max_tokens)
    else:
        base_preds = generate_predictions_openai(test_rows, args.base_model, max_tokens=args.max_tokens)
    base_time = time.time() - t0

    base_rouge = _compute_rouge(base_preds, references)
    base_em = _compute_exact_match(base_preds, references)
    results["base"] = {
        "model": args.base_model,
        "rouge": base_rouge,
        "exact_match": base_em,
        "inference_time_s": round(base_time, 1),
    }

    # --- Fine-tuned model ---
    if args.finetuned_adapter or args.finetuned_model:
        ft_model = args.finetuned_model or args.base_model
        ft_label = f"{ft_model}" + (f" + {args.finetuned_adapter}" if args.finetuned_adapter else "")
        print(f"\n=== Fine-tuned model: {ft_label} ===")
        t0 = time.time()
        if args.backend == "mlx":
            ft_preds = generate_predictions_mlx(
                test_rows, ft_model, adapter_path=args.finetuned_adapter, max_tokens=args.max_tokens
            )
        else:
            ft_preds = generate_predictions_openai(test_rows, ft_model, max_tokens=args.max_tokens)
        ft_time = time.time() - t0

        ft_rouge = _compute_rouge(ft_preds, references)
        ft_em = _compute_exact_match(ft_preds, references)
        results["finetuned"] = {
            "model": ft_label,
            "rouge": ft_rouge,
            "exact_match": ft_em,
            "inference_time_s": round(ft_time, 1),
        }

        # --- Comparison ---
        delta = {}
        for metric in ["rouge1_f", "rouge2_f", "rougeL_f"]:
            base_val = base_rouge.get(metric, 0.0)
            ft_val = ft_rouge.get(metric, 0.0)
            delta[metric] = round(ft_val - base_val, 4)
        delta["exact_match"] = round(ft_em - base_em, 4)

        # Bootstrap significance on ROUGE-L
        base_rl_scores = []
        ft_rl_scores = []
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            for bp, fp, ref in zip(base_preds, ft_preds, references):
                base_rl_scores.append(scorer.score(ref, bp)["rougeL"].fmeasure)
                ft_rl_scores.append(scorer.score(ref, fp)["rougeL"].fmeasure)
        except ImportError:
            pass

        if base_rl_scores:
            significance = _paired_randomization_test(
                base_rl_scores,
                ft_rl_scores,
                n_resamples=args.resamples,
                seed=args.seed,
            )
            confidence_interval = _paired_bootstrap_ci(
                base_rl_scores,
                ft_rl_scores,
                n_resamples=args.resamples,
                seed=args.seed,
            )
        else:
            significance = {}
            confidence_interval = {}

        results["comparison"] = {
            "delta": delta,
            "significance": significance,
            "rougeL_delta_confidence_interval": confidence_interval,
        }

        print("\n=== Comparison ===")
        for k, v in delta.items():
            direction = "+" if v > 0 else ""
            print(f"  {k}: {direction}{v}")
        if significance:
            sig_str = "YES" if significance.get("significant") else "no"
            print(f"  Significance (p<0.05): {sig_str} (p={significance.get('p_value', 'N/A')})")

    # Save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nResults saved to {out_path}")

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark base vs fine-tuned model.")
    ap.add_argument("--test-data", required=True, help="Test set JSONL")
    ap.add_argument("--base-model", required=True, help="Base model (HF ID or OpenAI model name)")
    ap.add_argument("--finetuned-model", default=None, help="Fine-tuned model (HF ID or OpenAI model)")
    ap.add_argument("--finetuned-adapter", default=None, help="Path to LoRA adapter directory (MLX)")
    ap.add_argument("--backend", choices=["mlx", "openai"], default="mlx", help="Inference backend")
    ap.add_argument("--max-tokens", type=int, default=256, help="Max generation tokens")
    ap.add_argument("--max-examples", type=int, default=None, help="Limit test examples")
    ap.add_argument("--resamples", type=int, default=5000, help="Statistical resamples")
    ap.add_argument("--seed", type=int, default=42, help="Statistical random seed")
    ap.add_argument("--output", default=None, help="Output path for comparison JSON")
    args = ap.parse_args()

    run_benchmark(args)


if __name__ == "__main__":
    main()
