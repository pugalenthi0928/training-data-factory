import json
import subprocess
import sys
from pathlib import Path


def write_jsonl(p: Path, rows):
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def read_jsonl(p: Path):
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def test_hash_dedupe(tmp_path: Path):
    inp = tmp_path / "data.jsonl"
    outp = tmp_path / "out.jsonl"
    rows = [
        {"output_text": "Paris is the capital of France."},
        {"output_text": "The capital of France is Paris."},
        {"output_text": "PARIS is the capital of France"},
        {"output_text": "Blue is a color."},
    ]
    write_jsonl(inp, rows)
    r = subprocess.run(
        [sys.executable, "scripts/compute_dedupe.py", "--input", str(inp), "--output", str(outp), "--method", "hash"],
        capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    kept = read_jsonl(outp)
    assert len(kept) == 2
