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


def normalise_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "task_name" not in df.columns:
        df["task_name"] = ""

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

    return df


def export_finetune(
    df: pd.DataFrame,
    task_name: Optional[str],
    out_path: Path,
    fmt: str,
    max_examples: Optional[int] = None,
) -> Dict[str, Any]:
    df = normalise_schema(df)

    if task_name:
        df = df[df["task_name"] == task_name]

    df = df.copy()

    df["input_text"] = df["input_text"].astype(str)
    df["output_text"] = df["output_text"].astype(str)

    df = df[(df["input_text"].str.strip() != "") & (df["output_text"].str.strip() != "")]

    if df.empty:
        raise SystemExit("No rows available after filtering and cleaning for export.")

    if max_examples is not None and max_examples > 0:
        df = df.head(max_examples)

    records: List[Dict[str, Any]] = []

    if fmt == "text":
        for _, row in df.iterrows():
            records.append(
                {
                    "input": row["input_text"],
                    "output": row["output_text"],
                }
            )
    elif fmt == "chat":
        for _, row in df.iterrows():
            records.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": str(row["input_text"]),
                        },
                        {
                            "role": "assistant",
                            "content": str(row["output_text"]),
                        },
                    ]
                }
            )
    else:
        raise SystemExit(f"Unsupported format: {fmt}")

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "input_rows": int(df.shape[0]),
        "exported_rows": len(records),
        "task_name": task_name,
        "format": fmt,
        "output_path": str(out_path),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a dataset JSONL into a fine-tuning format.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to source JSONL dataset (e.g. output/dataset_cli_rich_200.jsonl)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSONL for fine-tuning.",
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="Optional task_name to filter on (e.g. summary_v1, qa_v1).",
    )
    parser.add_argument(
        "--format",
        choices=["text", "chat"],
        default="text",
        help="Export format: 'text' (input/output fields) or 'chat' (messages[]).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on number of examples to export.",
    )

    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    df = load_jsonl(in_path)
    if df.empty:
        raise SystemExit("Input dataset is empty; nothing to export.")

    summary = export_finetune(
        df=df,
        task_name=args.task_name,
        out_path=out_path,
        fmt=args.format,
        max_examples=args.max_examples,
    )

    print("Fine-tune export summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
