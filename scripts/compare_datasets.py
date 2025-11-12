from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_jsonl(path: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records)


def normalise_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # input_text
    if "input_text" not in df.columns:
        if "input" in df.columns:
            df["input_text"] = df["input"].astype(str)
        elif "context" in df.columns:
            df["input_text"] = df["context"].astype(str)
        else:
            df["input_text"] = ""

    # output_text
    if "output_text" not in df.columns:
        if "output" in df.columns:
            df["output_text"] = df["output"].astype(str)
        elif "answer" in df.columns:
            df["output_text"] = df["answer"].astype(str)
        else:
            df["output_text"] = ""

    # task_name
    if "task_name" not in df.columns:
        if "question" in df.columns and "answer" in df.columns:
            df["task_name"] = "rag_qa"
        else:
            df["task_name"] = ""

    # task_type
    if "task_type" not in df.columns:
        if "question" in df.columns and "answer" in df.columns:
            df["task_type"] = "qa"
        else:
            df["task_type"] = ""

    if "document_id" not in df.columns:
        df["document_id"] = ""

    df["input_text"] = df["input_text"].astype(str)
    df["output_text"] = df["output_text"].astype(str)
    df["task_name"] = df["task_name"].astype(str)
    df["document_id"] = df["document_id"].astype(str)

    df["input_length"] = df["input_text"].str.len()
    df["output_length"] = df["output_text"].str.len()

    return df


def compute_stats(df: pd.DataFrame) -> Dict[str, Any]:
    num_examples = int(df.shape[0])
    num_documents = int(df["document_id"].nunique()) if "document_id" in df.columns else 0

    avg_input_len = float(df["input_length"].mean()) if num_examples > 0 else 0.0
    avg_output_len = float(df["output_length"].mean()) if num_examples > 0 else 0.0

    per_task_counts = (
        df["task_name"].value_counts()
        .to_dict()
    )

    return {
        "num_examples": num_examples,
        "num_documents": num_documents,
        "avg_input_length": avg_input_len,
        "avg_output_length": avg_output_len,
        "per_task_counts": per_task_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple dataset JSONL files and summarise basic stats."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more JSONL dataset paths.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write comparison JSON (summary).",
    )

    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    for p in input_paths:
        if not p.exists():
            raise SystemExit(f"Input file not found: {p}")

    summary: Dict[str, Any] = {"datasets": {}}

    for p in input_paths:
        df = load_jsonl(p)
        if df.empty:
            stats = {
                "num_examples": 0,
                "num_documents": 0,
                "avg_input_length": 0.0,
                "avg_output_length": 0.0,
                "per_task_counts": {},
            }
        else:
            df_norm = normalise_schema(df)
            stats = compute_stats(df_norm)

        label = p.name
        summary["datasets"][label] = {
            "path": str(p),
            **stats,
        }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Dataset comparison summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
