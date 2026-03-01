import argparse
import json
import sys
from pathlib import Path

from rouge_score import rouge_scorer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from training_data_robo.io import load_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Predictions (or dataset) JSONL")
    parser.add_argument("--output", default=None, help="Where to write metrics JSON")
    parser.add_argument(
        "--reference-column",
        default="output_text",
        help="Column name for reference/ground-truth text (default: output_text)",
    )
    parser.add_argument(
        "--prediction-column",
        default="output_text",
        help="Column name for predicted text (default: output_text)",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    rows = load_jsonl(in_path)
    if not rows:
        raise SystemExit("No rows to evaluate.")

    ref_col = args.reference_column
    pred_col = args.prediction_column

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1_f, r2_f, rl_f, em = [], [], [], []

    for row in rows:
        ref = str(row.get(ref_col, "") or "")
        pred = str(row.get(pred_col, "") or "")
        # ROUGE
        scores = scorer.score(ref, pred)
        r1_f.append(scores["rouge1"].fmeasure)
        r2_f.append(scores["rouge2"].fmeasure)
        rl_f.append(scores["rougeL"].fmeasure)
        # Exact match (strict, whitespace-trim)
        em.append(1.0 if pred.strip() == ref.strip() else 0.0)

    metrics = {
        "num_eval_examples": len(rows),
        "rouge1_f": float(sum(r1_f) / len(r1_f)),
        "rouge2_f": float(sum(r2_f) / len(r2_f)),
        "rougeL_f": float(sum(rl_f) / len(rl_f)),
        "exact_match": float(sum(em) / len(em)),
        "reference_column": ref_col,
        "prediction_column": pred_col,
    }

    out_path = Path(args.output) if args.output else None
    if out_path:
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    else:
        print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
