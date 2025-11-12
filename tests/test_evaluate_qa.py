import json
import subprocess
import sys
from pathlib import Path


def test_evaluate_qa_script(tmp_path: Path) -> None:
    """Smoke test: evaluate_qa.py should run and produce metrics JSON."""
    data_path = tmp_path / "qa_dataset.jsonl"
    metrics_path = tmp_path / "metrics.json"

    rows = [
        {
            "task_type": "qa",
            "answer": "Paris is the capital of France.",
            "output_text": "Paris is the capital of France.",
        },
        {
            "task_type": "qa",
            "answer": "Blue",
            "output_text": "Red",
        },
    ]

    with data_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_qa.py",
            "--input",
            str(data_path),
            "--output",
            str(metrics_path),
        ],
        capture_output=True,
        text=True,
    )

    # Script should exit cleanly
    assert result.returncode == 0, result.stderr

    # Metrics file should exist and be valid JSON
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Basic sanity checks on keys / ranges
    assert metrics["num_eval_examples"] == 2
    assert 0.0 <= metrics["exact_match"] <= 1.0
    assert 0.0 <= metrics["rouge1_f"] <= 1.0
    assert 0.0 <= metrics["rouge2_f"] <= 1.0
    assert 0.0 <= metrics["rougeL_f"] <= 1.0
