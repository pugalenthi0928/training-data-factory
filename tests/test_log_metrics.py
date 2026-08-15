import csv
import json
import subprocess
import sys
from pathlib import Path


def test_log_metrics(tmp_path: Path):
    metrics = tmp_path / "m.json"
    dataset = tmp_path / "d.jsonl"
    csvp = tmp_path / "runs.csv"

    metrics.write_text(
        json.dumps(
            {
                "num_eval_examples": 5,
                "rouge1_f": 0.5,
                "rouge2_f": 0.3,
                "rougeL_f": 0.4,
                "exact_match": 0.2,
            }
        ),
        encoding="utf-8",
    )
    dataset.write_text('{"x":1}\n{"x":2}\n', encoding="utf-8")

    r = subprocess.run(
        [
            sys.executable,
            "scripts/log_metrics.py",
            "--metrics",
            str(metrics),
            "--dataset",
            str(dataset),
            "--model",
            "fake-1",
            "--csv",
            str(csvp),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert csvp.exists()

    with csvp.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2  # header + 1 row
    assert rows[1][1] == "fake-1"
