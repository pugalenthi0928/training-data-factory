from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def test_export_finetune_text(tmp_path: Path) -> None:
    data_path = tmp_path / "dataset.jsonl"
    out_path = tmp_path / "finetune_text.jsonl"

    rows = [
        {
            "task_name": "summary_v1",
            "input_text": "This is the first document.",
            "output_text": "A short summary.",
        },
        {
            "task_name": "qa_v1",
            "input_text": "Some context here.",
            "output_text": "An answer.",
        },
    ]
    write_jsonl(data_path, rows)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_finetune.py",
            "--input",
            str(data_path),
            "--output",
            str(out_path),
            "--task-name",
            "summary_v1",
            "--format",
            "text",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert out_path.exists()

    exported = read_jsonl(out_path)
    assert len(exported) == 1
    row = exported[0]
    assert set(row.keys()) == {"input", "output"}
    assert row["input"] == "This is the first document."
    assert row["output"] == "A short summary."


def test_export_rag_qa(tmp_path: Path) -> None:
    data_path = tmp_path / "qa_dataset.jsonl"
    out_path = tmp_path / "rag_qa.jsonl"

    rows = [
        {
            "task_name": "qa_v1",
            "question": "What is the capital of France?",
            "answer": "Paris.",
            "context": "France's capital city is Paris.",
            "document_id": "doc1",
            "chunk_id": "c1",
        },
        {
            "task_name": "summary_v1",
            "question": "",
            "answer": "",
            "context": "",
            "document_id": "doc2",
            "chunk_id": "c2",
        },
    ]
    write_jsonl(data_path, rows)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/export_rag_qa.py",
            "--input",
            str(data_path),
            "--output",
            str(out_path),
            "--task-name",
            "qa_v1",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert out_path.exists()

    exported = read_jsonl(out_path)
    assert len(exported) == 1
    row = exported[0]
    assert row["question"] == "What is the capital of France?"
    assert row["answer"] == "Paris."
    assert row["document_id"] == "doc1"
    assert row["chunk_id"] == "c1"
    assert row["context"] == "France's capital city is Paris."
