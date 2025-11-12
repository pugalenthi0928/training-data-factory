from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any


def test_postprocess_quality_script(tmp_path: Path) -> None:
    data_path = tmp_path / "dataset.jsonl"
    out_path = tmp_path / "dataset_quality.jsonl"

    rows: List[Dict[str, Any]] = [
        {
            "task_name": "qa_v1",
            "task_type": "qa",
            "question": "What is the capital of France?",
            "answer": "Paris is the capital of France.",
            "output_text": "Paris is the capital of France.",
            "context": "France's capital city is Paris.",
        },
        {
            "task_name": "qa_v1",
            "task_type": "qa",
            "question": "What is 2+2?",
            "answer": "4",
            "output_text": "As an AI language model, I cannot answer that.",
            "context": "2+2 equals 4.",
        },
    ]

    with data_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/postprocess_quality.py",
            "--input",
            str(data_path),
            "--output",
            str(out_path),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert out_path.exists()

    processed: List[Dict[str, Any]] = []
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                processed.append(json.loads(line))

    assert len(processed) == 2

    for row in processed:
        assert "quality_score" in row
        assert "quality_flags" in row
        assert isinstance(row["quality_flags"], list)

    flags_second = processed[1]["quality_flags"]
    assert any("refusal" in f for f in flags_second)
