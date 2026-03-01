#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from training_data_robo.io import count_jsonl_rows


def sha256_file(p: Path, chunk=1<<20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def quality_summary(p: Path) -> Dict[str, Any]:
    total = 0
    score_sum = 0.0
    flag_counts: Dict[str,int] = {}
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                total += 1
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                s = row.get("quality_score")
                if isinstance(s,(int,float)): score_sum += float(s)
                for fl in row.get("quality_flags",[]) or []:
                    flag_counts[fl] = flag_counts.get(fl,0)+1
        avg = (score_sum/total) if total else None
        return {"rows_scored": total, "avg_quality_score": avg, "flag_counts": flag_counts}
    except FileNotFoundError:
        return {"rows_scored": 0, "avg_quality_score": None, "flag_counts": {}}

def load_json(p: Path):
    if not p.exists(): return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description="Register a dataset (manifest entry with stats, hashes, metrics).")
    ap.add_argument("--dataset-path", required=True, help="Final deduped dataset (.jsonl)")
    ap.add_argument("--raw-dataset", default=None, help="Pre-dedup dataset (.jsonl) to compute drop delta")
    ap.add_argument("--metrics", default=None, help="Evaluation metrics JSON (optional)")
    ap.add_argument("--name", required=True, help="Dataset logical name (e.g., papers_qa)")
    ap.add_argument("--version", required=True, help="Dataset version (e.g., 0.8.0)")
    ap.add_argument("--source-tag", default="", help="Freeform tag for provenance (e.g., papers_qa)")
    ap.add_argument("--model", default="", help="Model used for predictions/eval (e.g., gpt-4.1-mini)")
    ap.add_argument("--registry", default="registry/manifest.json", help="Manifest path")
    args = ap.parse_args()

    outp = Path(args.dataset_path)
    rawp = Path(args.raw_dataset) if args.raw_dataset else None
    metp = Path(args.metrics) if args.metrics else None
    manp = Path(args.registry)

    outp.parent.mkdir(parents=True, exist_ok=True)
    manp.parent.mkdir(parents=True, exist_ok=True)

    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    num_rows = count_jsonl_rows(outp)
    raw_rows = count_jsonl_rows(rawp) if rawp and rawp.exists() else None
    dropped = (raw_rows - num_rows) if (raw_rows is not None) else None

    qsum = quality_summary(outp)
    metrics = load_json(metp) if metp else None

    record = {
        "timestamp": now,
        "name": args.name,
        "version": args.version,
        "source_tag": args.source_tag,
        "paths": {
            "dataset": str(outp),
            "raw_dataset": str(rawp) if rawp else None,
            "metrics": str(metp) if metp else None,
        },
        "hashes": {
            "dataset_sha256": sha256_file(outp),
            "raw_sha256": sha256_file(rawp) if rawp and rawp.exists() else None,
        },
        "counts": {
            "rows": num_rows,
            "raw_rows": raw_rows,
            "dedupe_dropped": dropped,
        },
        "quality": qsum,
        "metrics": metrics,
        "model": args.model,
        "env": {
            "user": os.environ.get("USER") or os.environ.get("USERNAME"),
            "hostname": os.uname().nodename if hasattr(os, "uname") else None,
        },
    }

    manifest = []
    if manp.exists():
        try:
            manifest = json.loads(manp.read_text(encoding="utf-8"))
            if not isinstance(manifest, list): manifest = []
        except Exception:
            manifest = []
    manifest.append(record)
    manp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"registered": True, "registry": str(manp), "added_record": record}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
