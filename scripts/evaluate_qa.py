from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from rouge_score import rouge_scorer

def load_jsonl(p: Path) -> pd.DataFrame:
    recs: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                recs.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(recs) if recs else pd.DataFrame()

def normalise_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "input_text" not in df.columns:
        if "input" in df.columns:
            df["input_text"] = df["input"].astype(str)
        elif "context" in df.columns:
            df["input_text"] = df["context"].astype(str)
        else:
            df["input_text"] = ""
    if "output_text" not in df.columns:
        if "output" in df.columns:
            df["output_text"] = df["output"].astype(str)
        elif "answer" in df.columns:
            df["output_text"] = df["answer"].astype(str)
        else:
            df["output_text"] = ""

    for c in ("task_name","task_type","question","answer","context","prediction"):
        if c not in df.columns:
            df[c] = ""
    return df

def choose_columns(df: pd.DataFrame) -> tuple[str, str, pd.DataFrame]:
    # Reference (ground truth)
    ref_col = "answer" if df["answer"].astype(str).str.strip().any() else "output_text"
    # Prediction (model output)
    pred_col = "prediction" if df["prediction"].astype(str).str.strip().any() else "output_text"

    # Filter QA rows
    mask = (
        df["task_type"].astype(str).str.contains("qa", case=False, na=False)
        | df["task_name"].astype(str).str.contains("qa", case=False, na=False)
        | (df["question"].astype(str).str.strip() != "")
    )
    qa_df = df[mask].copy()
    qa_df = qa_df[(qa_df[ref_col].astype(str).str.strip() != "") & (qa_df[pred_col].astype(str).str.strip() != "")]
    return ref_col, pred_col, qa_df

def exact_match(a: str, b: str) -> float:
    na = " ".join(a.split()).strip().lower()
    nb = " ".join(b.split()).strip().lower()
    return 1.0 if na == nb else 0.0

def compute_qa_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        raise SystemExit("Dataset is empty; nothing to evaluate.")

    df = normalise_schema(df)
    ref_col, pred_col, qa_df = choose_columns(df)
    if qa_df.empty:
        raise SystemExit("No QA examples found (need reference and prediction columns).")

    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)

    n = 0
    r1_f = r2_f = rL_f = em = 0.0
    for _, row in qa_df.iterrows():
        ref = str(row[ref_col])
        pred = str(row[pred_col])
        scores = scorer.score(ref, pred)
        r1_f += scores["rouge1"].fmeasure
        r2_f += scores["rouge2"].fmeasure
        rL_f += scores["rougeL"].fmeasure
        em += exact_match(ref, pred)
        n += 1

    if n == 0:
        raise SystemExit("No comparable QA rows after filtering.")

    metrics = {
        "num_eval_examples": n,
        "rouge1_f": r1_f / n,
        "rouge2_f": r2_f / n,
        "rougeL_f": rL_f / n,
        "exact_match": em / n,
        "reference_column": ref_col,
        "prediction_column": pred_col,
    }
    return metrics

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate QA dataset with ROUGE + EM.")
    ap.add_argument("--input", required=True, help="Input JSONL path.")
    ap.add_argument("--output", help="Optional path to write metrics JSON.")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    df = load_jsonl(in_path)
    metrics = compute_qa_metrics(df)
    print("QA evaluation metrics:")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.output:
        Path(args.output).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nMetrics written to: {args.output}")

if __name__ == "__main__":
    main()
