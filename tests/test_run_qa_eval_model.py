import json
import subprocess
import sys
from pathlib import Path


def write_jsonl(p: Path, rows):
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def read_jsonl(p: Path):
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def test_fake_run_and_eval(tmp_path: Path):
    data = tmp_path / "qa.jsonl"
    preds = tmp_path / "preds.jsonl"
    metrics = tmp_path / "metrics.json"

    rows = [
        {"task_name":"qa_v1","question":"What is the capital of France?","answer":"Paris","context":"Paris is the capital of France."},
        {"task_name":"qa_v1","question":"Favorite color?","answer":"Blue","context":"Her favorite color is Blue."},
    ]
    write_jsonl(data, rows)

    # Generate predictions (fake model)
    r = subprocess.run(
        [sys.executable, "scripts/run_qa_eval_model.py", "--input", str(data), "--output", str(preds), "--fake-model"],
        capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    assert preds.exists()

    pred_rows = read_jsonl(preds)
    assert len(pred_rows) == 2
    assert all("prediction" in r for r in pred_rows)

    # Evaluate (should be perfect with fake model)
    r2 = subprocess.run(
        [sys.executable, "scripts/evaluate_qa.py", "--input", str(preds), "--output", str(metrics)],
        capture_output=True, text=True
    )
    assert r2.returncode == 0, r2.stderr
    assert metrics.exists()
    m = json.loads(metrics.read_text(encoding="utf-8"))
    assert m["num_eval_examples"] == 2
    assert m["exact_match"] == 1.0
