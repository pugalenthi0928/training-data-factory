from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Dict, Any


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a fine-tuning dataset from a Training Data Robo JSONL file."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input JSONL dataset (e.g. output/papers_rich_summary_qa_300.jsonl)",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output JSONL path for fine-tune format.",
    )
    parser.add_argument(
        "-t", "--task-name",
        required=True,
        help="Task name to keep, e.g. summary_v1 or qa_v1.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for rec in read_jsonl(in_path):
            if rec.get("task_name") != args.task_name:
                continue

            inp = rec.get("input_text", "")
            out = rec.get("output_text", "")

            new_rec = {
                "input": inp,
                "output": out,
                # keep some helpful metadata
                "task_name": rec.get("task_name"),
                "task_type": rec.get("task_type"),
                "document_id": rec.get("document_id"),
                "chunk_id": rec.get("chunk_id"),
                "model_name": rec.get("model_name"),
                "task_version": rec.get("task_version"),
            }
            out_f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote {kept} examples to {out_path}")


if __name__ == "__main__":
    main()
