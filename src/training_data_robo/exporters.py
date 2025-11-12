from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .logging_config import get_logger

logger = get_logger("training_data_robo.exporters")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line in %s", path)
                continue
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    """Write an iterable of dicts as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %s", path)


# ---------- Fine-tuning exporter ----------


def export_finetune(
    input_path: Path,
    output_path: Path,
    fmt: str = "chat",
) -> None:
    """
    Convert a Training Data Robo dataset JSONL into a fine-tuning format.

    Supported formats:
      - "chat":  {"messages":[{role,content}...]}  (OpenAI-style)
      - "io":    {"input": "...", "output": "..."} (generic)
    """
    if fmt not in {"chat", "io"}:
        raise ValueError(f"Unsupported format: {fmt!r}")

    records = load_jsonl(input_path)
    logger.info("Loaded %d records from %s", len(records), input_path)

    out: List[Dict[str, Any]] = []

    for rec in records:
        inp = str(rec.get("input_text", "") or "").strip()
        out_text = str(rec.get("output_text", "") or "").strip()
        if not inp or not out_text:
            continue

        task_name = str(rec.get("task_name", "") or "")
        system_content = f"You are a helpful assistant. Task: {task_name}".strip()

        if fmt == "chat":
            ft_example = {
                "messages": [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": inp},
                    {"role": "assistant", "content": out_text},
                ]
            }
        else:  # fmt == "io"
            ft_example = {
                "input": inp,
                "output": out_text,
                "task_name": task_name,
            }

        out.append(ft_example)

    write_jsonl(output_path, out)
    logger.info(
        "Exported %d fine-tune examples (%s) from %s to %s",
        len(out),
        fmt,
        input_path,
        output_path,
    )


# ---------- RAG QA exporter ----------


def _parse_qa_output(text: str) -> Optional[Tuple[str, str]]:
    """
    Parse a 'Question: ... Answer: ...' style output into (question, answer).
    Returns None if parsing fails.
    """
    if not text:
        return None

    lower = text.lower()
    q_idx = lower.find("question:")
    a_idx = lower.find("answer:")

    if q_idx == -1 or a_idx == -1 or a_idx <= q_idx:
        return None

    q_part = text[q_idx + len("Question:") : a_idx]
    a_part = text[a_idx + len("Answer:") :]

    question = q_part.strip(" \n:").strip()
    answer = a_part.strip()

    if not question or not answer:
        return None

    return question, answer


def _extract_passage_from_input(input_text: str) -> str:
    """
    For QA tasks, we usually embed the passage like:

      'Read the following passage and generate ...\n\nPassage:\n\n{chunk_text}'

    This helper pulls out {chunk_text}. If the pattern isn't found,
    we just return the full input_text.
    """
    if not input_text:
        return ""

    marker = "\n\nPassage:\n\n"
    idx = input_text.find(marker)
    if idx == -1:
        return input_text.strip()

    return input_text[idx + len(marker) :].strip()


def export_rag_qa(
    input_path: Path,
    output_path: Path,
    qa_task_name: str = "qa_v1",
) -> None:
    """
    Convert a QA-style dataset into a RAG-friendly JSONL:

      {
        "question": "...",
        "answer": "...",
        "context": "...",
        "document_id": "...",
        "chunk_id": "...",
        "task_name": "...",
        "model_name": "...",
        "metadata": {...}
      }

    It uses:
      - question/answer parsed from output_text
      - context extracted from input_text (the passage)
    """
    records = load_jsonl(input_path)
    logger.info("Loaded %d records from %s", len(records), input_path)

    out: List[Dict[str, Any]] = []
    skipped = 0

    for rec in records:
        if str(rec.get("task_name", "")) != qa_task_name:
            continue

        output_text = str(rec.get("output_text", "") or "")
        parsed = _parse_qa_output(output_text)
        if not parsed:
            skipped += 1
            continue

        question, answer = parsed

        input_text = str(rec.get("input_text", "") or "")
        context = _extract_passage_from_input(input_text)

        out_rec: Dict[str, Any] = {
            "question": question,
            "answer": answer,
            "context": context,
            "document_id": rec.get("document_id"),
            "chunk_id": rec.get("chunk_id"),
            "task_name": rec.get("task_name"),
            "model_name": rec.get("model_name"),
            "metadata": rec.get("metadata", {}),
        }
        out.append(out_rec)

    write_jsonl(output_path, out)
    logger.info(
        "Exported %d RAG QA examples from %s to %s (skipped %d)",
        len(out),
        input_path,
        output_path,
        skipped,
    )
