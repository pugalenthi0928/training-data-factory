from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from rouge_score import rouge_scorer


def load_jsonl(path: Path) -> pd.DataFrame:
    """Simple JSONL loader (mirrors app.py logic)."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines
                continue
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def normalise_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise to input_text / output_text / task_name / task_type, like app.py."""
    df = df.copy()

    # input_text
    if "input_text" not in df.columns:
        if "input" in df.columns:
            df["input_text"] = df["input"].astype(str)
        elif "question" in df.columns or "context" in df.columns:
            # For RAG-style datasets: prefer context, fall back to question
            if "context" in df.columns:
                df["input_text"] = df["context"].astype(str)
            else:
                df["input_text"] = df["question"].astype(str)
        else:
            df["input_text"] = ""

    # output_text
    if "output_text" not in df.columns:
        if "output" in df.columns:
            df["output_text"] = df["output"].astype(str)
        elif "answer" in df.columns:
            # For QA datasets with an explicit gold answer
            df["output_text"] = df["answer"].astype(str)
        else:
            df["output_text"] = ""

    # task_name default
    if "task_name" not in df.columns:
        if "question" in df.columns and "answer" in df.columns:
            df["task_name"] = "qa"
        else:
            df["task_name"] = ""

    # task_type default (not strictly needed, but kept for consistency)
    if "task_type" not in df.columns:
        if "question" in df.columns and "answer" in df.columns:
            df["task_type"] = "qa"
        else:
            df["task_type"] = ""

    return df


def compute_qa_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute ROUGE + exact match for QA examples.

    - Detect QA rows using task_name and/or task_type.
    - Choose reference and prediction columns based on what exists.
    """
    df = df.copy()

    # 1) Restrict to QA rows using OR over task_name / task_type
    if "task_name" in df.columns or "task_type" in df.columns:
        qa_mask = pd.Series(False, index=df.index)

        if "task_name" in df.columns:
            qa_mask |= df["task_name"].astype(str).str.contains(
                "qa", case=False, na=False
            )

        if "task_type" in df.columns:
            qa_mask |= df["task_type"].astype(str).str.lower().eq("qa")

        df = df[qa_mask]
    else:
        # No task_name/task_type columns -> treat all rows as QA
        pass

    if df.empty:
        raise SystemExit(
            "No QA examples found (no rows flagged as QA via task_name/task_type)."
        )

    # 2) Decide reference and prediction columns
    # Reference: gold answer
    if "answer" in df.columns:
        ref_col = "answer"
    else:
        # Fallback: assume output_text is the gold answer (TDR-style datasets)
        ref_col = "output_text"

    # Prediction: model output
    if "prediction" in df.columns:
        pred_col = "prediction"
    elif "model_output" in df.columns:
        pred_col = "model_output"
    elif "output_text" in df.columns:
        # Fallback: use output_text as prediction as well (self-consistency mode)
        pred_col = "output_text"
    else:
        raise SystemExit(
            "No suitable prediction column found (expected 'prediction', 'model_output' or 'output_text')."
        )

    df = df.dropna(subset=[ref_col, pred_col])
    if df.empty:
        raise SystemExit("No non-empty QA examples to evaluate after filtering.")

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    totals = {
        "rouge1": 0.0,
        "rouge2": 0.0,
        "rougeL": 0.0,
        "em": 0.0,
    }
    n = 0

    for _, row in df.iterrows():
        ref = str(row[ref_col]).strip()
        pred = str(row[pred_col]).strip()
        if not ref or not pred:
            continue

        scores = scorer.score(ref, pred)
        totals["rouge1"] += scores["rouge1"].fmeasure
        totals["rouge2"] += scores["rouge2"].fmeasure
        totals["rougeL"] += scores["rougeL"].fmeasure

        # Simple exact match on lowercased strings
        totals["em"] += 1.0 if ref.lower() == pred.lower() else 0.0
        n += 1

    if n == 0:
        raise SystemExit("After filtering, no QA examples remained to evaluate.")

    metrics: Dict[str, Any] = {
        "num_eval_examples": int(n),
        "rouge1_f": totals["rouge1"] / n,
        "rouge2_f": totals["rouge2"] / n,
        "rougeL_f": totals["rougeL"] / n,
        "exact_match": totals["em"] / n,
        "reference_column": ref_col,
        "prediction_column": pred_col,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate QA-style datasets with ROUGE + exact match."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSONL dataset file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write metrics JSON. If omitted, only prints to stdout.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    df = load_jsonl(in_path)
    if df.empty:
        raise SystemExit("Dataset is empty; nothing to evaluate.")

    df = normalise_schema(df)
    metrics = compute_qa_metrics(df)

    print("QA evaluation metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nMetrics written to: {out_path}")


if __name__ == "__main__":
    main()
