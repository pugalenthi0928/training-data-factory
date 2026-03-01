from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser(description="Append QA metrics to leaderboard CSV.")
    ap.add_argument("--metrics", required=True, help="Path to metrics JSON (from evaluate_qa.py)")
    ap.add_argument("--dataset", required=True, help="Path to dataset used for evaluation")
    ap.add_argument("--model", required=True, help="Model name used for predictions")
    ap.add_argument("--csv", default="runs/qa_runs.csv", help="Leaderboard CSV path (default: runs/qa_runs.csv)")
    args = ap.parse_args()

    mpath = Path(args.metrics)
    dpath = Path(args.dataset)
    cpath = Path(args.csv)

    metrics = json.loads(mpath.read_text(encoding="utf-8"))
    dataset_hash = sha256_file(dpath)

    cpath.parent.mkdir(parents=True, exist_ok=True)
    header = ["timestamp","model","dataset","dataset_sha256","num_eval_examples","rouge1_f","rouge2_f","rougeL_f","exact_match","metrics_path"]
    write_header = not cpath.exists()

    with cpath.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow([
            datetime.utcnow().isoformat(timespec="seconds"),
            args.model,
            str(dpath),
            dataset_hash,
            metrics.get("num_eval_examples"),
            metrics.get("rouge1_f"),
            metrics.get("rouge2_f"),
            metrics.get("rougeL_f"),
            metrics.get("exact_match"),
            str(mpath),
        ])
    print(f"Wrote leaderboard row to: {cpath}")

if __name__ == "__main__":
    main()
