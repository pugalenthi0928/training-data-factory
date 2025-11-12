from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterator, Dict, Any, Tuple


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


def extract_context(input_text: str) -> str:
    """Grab just the passage from the prompt."""
    if not input_text:
        return ""
    marker = "Passage:\n\n"
    idx = input_text.find(marker)
    if idx != -1:
        return input_text[idx + len(marker) :].strip()
    return input_text.strip()


def parse_qa(text: str) -> Tuple[str, str]:
    """
    Parse 'Question: ... Answer: ...' style text into (question, answer).
    Falls back gracefully if the format is a bit different.
    """
    if not text:
        return "", ""

    # Try to capture question between 'Question:' and 'Answer:'
    q_match = re.search(
        r"(?is)question\s*:\s*(.+?)(?:\n\s*\n|answer\s*:)", text, flags=re.DOTALL
    )
    a_match = re.search(r"(?is)answer\s*:\s*(.+)", text, flags=re.DOTALL)

    if q_match:
        question = q_match.group(1).strip()
    else:
        # fallback: whole thing as question
        question = text.strip()

    if a_match:
        answer = a_match.group(1).strip()
    else:
        answer = ""

    return question, answer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a RAG-style QA dataset from a Training Data Robo JSONL file."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input JSONL dataset (e.g. output/papers_rich_summary_qa_300.jsonl)",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output JSONL path for RAG QA dataset.",
    )
    parser.add_argument(
        "-t", "--task-name",
        default="qa_v1",
        help="Task name to filter on (default: qa_v1).",
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

            context = extract_context(rec.get("input_text", ""))
            question, answer = parse_qa(rec.get("output_text", ""))

            if not question or not context:
                continue

            new_rec = {
                "question": question,
                "answer": answer,
                "context": context,
                "document_id": rec.get("document_id"),
                "chunk_id": rec.get("chunk_id"),
                "model_name": rec.get("model_name"),
                "task_name": rec.get("task_name"),
                "task_version": rec.get("task_version"),
            }
            out_f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Wrote {kept} QA examples to {out_path}")


if __name__ == "__main__":
    main()
