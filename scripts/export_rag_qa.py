from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def normalise_qa_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "task_name" not in df.columns:
        df["task_name"] = ""

    # Question
    if "question" not in df.columns:
        if "input_text" in df.columns:
            df["question"] = df["input_text"].astype(str)
        else:
            df["question"] = ""

    # Answer
    if "answer" not in df.columns:
        if "output_text" in df.columns:
            df["answer"] = df["output_text"].astype(str)
        elif "output" in df.columns:
            df["answer"] = df["output"].astype(str)
        else:
            df["answer"] = ""

    # Context
    if "context" not in df.columns:
        if "input_text" in df.columns:
            df["context"] = df["input_text"].astype(str)
        else:
            df["context"] = ""

    if "document_id" not in df.columns:
        df["document_id"] = ""
    if "chunk_id" not in df.columns:
        df["chunk_id"] = ""

    df["question"] = df["question"].astype(str)
    df["answer"] = df["answer"].astype(str)
    df["context"] = df["context"].astype(str)
    df["document_id"] = df["document_id"].astype(str)
    df["chunk_id"] = df["chunk_id"].astype(str)

    return df


def export_rag_qa(
    df: pd.DataFrame,
    out_path: Path,
    task_name: Optional[str] = None,
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    df = normalise_qa_schema(df)

    if task_name:
        df = df[df["task_name"] == task_name]

    df = df[
        (df["question"].str.strip() != "")
        & (df["answer"].str.strip() != "")
    ]

    if df.empty:
        raise SystemExit("No QA rows available for export (question/answer empty).")

    if max_examples is not None and max_examples > 0:
        df = df.head(max_examples)

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        records.append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "document_id": row["document_id"],
                "chunk_id": row["chunk_id"],
                "context": row["context"],
            }
        )

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "input_rows": int(df.shape[0]),
        "exported_rows": len(records),
        "task_name": task_name,
        "output_path": str(out_path),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export QA rows into a RAG-friendly schema."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to source JSONL dataset (e.g. output/papers_qa_only_real_gpt4.jsonl)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSONL (RAG QA dataset).",
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="Optional task_name filter (e.g. qa_v1).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on number of examples.",
    )

    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    df = load_jsonl(in_path)
    if df.empty:
        raise SystemExit("Input dataset is empty; nothing to export.")

    summary = export_rag_qa(
        df=df,
        out_path=out_path,
        task_name=args.task_name,
        max_examples=args.max_examples,
    )

    print("RAG QA export summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
