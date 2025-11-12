from __future__ import annotations

import argparse
import json
from collections import Counter
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
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def normalise_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Map to input_text
    if "input_text" not in df.columns:
        if "input" in df.columns:
            df["input_text"] = df["input"].astype(str)
        elif "context" in df.columns:
            df["input_text"] = df["context"].astype(str)
        elif "question" in df.columns:
            df["input_text"] = df["question"].astype(str)
        else:
            df["input_text"] = ""

    # Map to output_text
    if "output_text" not in df.columns:
        if "output" in df.columns:
            df["output_text"] = df["output"].astype(str)
        elif "answer" in df.columns:
            df["output_text"] = df["answer"].astype(str)
        else:
            df["output_text"] = ""

    # Task name / type defaults
    if "task_name" not in df.columns:
        if "question" in df.columns and "answer" in df.columns:
            df["task_name"] = "rag_qa"
        else:
            df["task_name"] = ""

    if "task_type" not in df.columns:
        if "question" in df.columns and "answer" in df.columns:
            df["task_type"] = "qa"
        else:
            df["task_type"] = ""

    # Ensure some expected columns exist
    for col in ["document_id", "model_name", "task_version"]:
        if col not in df.columns:
            df[col] = ""

    # Synthetic IDs if missing/empty
    if "id" not in df.columns or df["id"].astype(str).str.strip().eq("").all():
        df["id"] = [f"ex-{i:05d}" for i in range(1, len(df) + 1)]

    # Lengths
    df["input_length"] = df["input_text"].astype(str).str.len()
    df["output_length"] = df["output_text"].astype(str).str.len()

    return df


def make_card(df: pd.DataFrame, dataset_name: str) -> str:
    num_examples = len(df)
    task_counts = Counter(df["task_name"].astype(str))
    type_counts = Counter(df["task_type"].astype(str))

    num_docs = df["document_id"].astype(str).nunique() if "document_id" in df.columns else 0
    avg_in = float(df["input_length"].mean()) if num_examples else 0.0
    avg_out = float(df["output_length"].mean()) if num_examples else 0.0

    lines: List[str] = []
    lines.append(f"# Dataset Card – {dataset_name}")
    lines.append("")
    lines.append("## 1. Summary")
    lines.append("")
    lines.append(f"- **Total examples:** {num_examples}")
    lines.append(f"- **Distinct tasks:** {len(task_counts)}")
    lines.append(f"- **Distinct task types:** {len(type_counts)}")
    lines.append(f"- **Distinct documents:** {num_docs}")
    lines.append(f"- **Avg input length (chars):** {avg_in:.1f}")
    lines.append(f"- **Avg output length (chars):** {avg_out:.1f}")
    lines.append("")

    lines.append("## 2. Task Breakdown")
    lines.append("")
    if task_counts:
        lines.append("| task_name | count |")
        lines.append("|-----------|-------|")
        for name, count in task_counts.most_common():
            lines.append(f"| `{name}` | {count} |")
        lines.append("")
    else:
        lines.append("_No task_name information available._")
        lines.append("")

    lines.append("## 3. Task Type Breakdown")
    lines.append("")
    if type_counts:
        lines.append("| task_type | count |")
        lines.append("|-----------|-------|")
        for name, count in type_counts.most_common():
            lines.append(f"| `{name}` | {count} |")
        lines.append("")
    else:
        lines.append("_No task_type information available._")
        lines.append("")

    lines.append("## 4. Length Statistics")
    lines.append("")
    if num_examples:
        in_desc = df["input_length"].describe()
        out_desc = df["output_length"].describe()

        def fmt_desc(prefix: str, desc: pd.Series) -> List[str]:
            return [
                f"- **{prefix} min:** {desc['min']:.1f}",
                f"- **{prefix} 25%:** {desc['25%']:.1f}",
                f"- **{prefix} mean:** {desc['mean']:.1f}",
                f"- **{prefix} 75%:** {desc['75%']:.1f}",
                f"- **{prefix} max:** {desc['max']:.1f}",
            ]

        lines.append("### Input length")
        lines.extend(fmt_desc("Input", in_desc))
        lines.append("")
        lines.append("### Output length")
        lines.extend(fmt_desc("Output", out_desc))
        lines.append("")
    else:
        lines.append("_No length statistics; dataset is empty._")
        lines.append("")

    lines.append("## 5. Example Record")
    lines.append("")
    if num_examples:
        ex = df.iloc[0]
        lines.append("```json")
        example_dict = {
            "id": str(ex.get("id", "")),
            "task_name": str(ex.get("task_name", "")),
            "task_type": str(ex.get("task_type", "")),
            "model_name": str(ex.get("model_name", "")),
            "document_id": str(ex.get("document_id", "")),
            "input_text": str(ex.get("input_text", ""))[:500],
            "output_text": str(ex.get("output_text", ""))[:500],
        }
        lines.append(json.dumps(example_dict, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")
    else:
        lines.append("_Dataset is empty; no example to show._")
        lines.append("")

    lines.append("## 6. Usage Notes")
    lines.append("")
    lines.append("- Suitable for fine-tuning (input → output pairs).")
    lines.append("- Suitable for RAG QA (if question/answer/context fields exist).")
    lines.append("- Generated by Training Data Robo – LLM Training Data Factory.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Markdown dataset card from a JSONL dataset.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to JSONL dataset file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output Markdown file.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    df = load_jsonl(in_path)
    if df.empty:
        raise SystemExit("Dataset is empty; nothing to summarise.")

    df = normalise_schema(df)
    card = make_card(df, dataset_name=in_path.name)

    out_path.write_text(card, encoding="utf-8")
    print(f"Wrote dataset card to: {out_path}")


if __name__ == "__main__":
    main()
