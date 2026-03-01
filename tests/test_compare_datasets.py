from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_compare_datasets(tmp_path: Path) -> None:
    data1 = tmp_path / "d1.jsonl"
    data2 = tmp_path / "d2.jsonl"
    out_path = tmp_path / "comparison.json"

    rows1 = [
        {
            "task_name": "summary_v1",
            "input_text": "Doc one.",
            "output_text": "Summary one.",
            "document_id": "doc1",
        },
        {
            "task_name": "summary_v1",
            "input_text": "Doc two.",
            "output_text": "Summary two.",
            "document_id": "doc1",
        },
    ]
    rows2 = [
        {
            "task_name": "qa_v1",
            "input_text": "Context here.",
            "output_text": "Answer here.",
            "document_id": "doc2",
        }
    ]

    write_jsonl(data1, rows1)
    write_jsonl(data2, rows2)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_datasets.py",
            "--inputs",
            str(data1),
            str(data2),
            "--output",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert out_path.exists()

    summary = json.loads(out_path.read_text(encoding="utf-8"))
    assert "datasets" in summary
    assert set(summary["datasets"].keys()) == {"d1.jsonl", "d2.jsonl"}

    d1_stats = summary["datasets"]["d1.jsonl"]
    assert d1_stats["num_examples"] == 2
    assert d1_stats["num_documents"] == 1
    assert d1_stats["per_task_counts"]["summary_v1"] == 2

    d2_stats = summary["datasets"]["d2.jsonl"]
    assert d2_stats["num_examples"] == 1
    assert d2_stats["num_documents"] == 1
    assert d2_stats["per_task_counts"]["qa_v1"] == 1
