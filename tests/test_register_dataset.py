import json
import subprocess
import sys
from pathlib import Path


def write_jsonl(p: Path, rows):
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_register(tmp_path: Path):
    dedup = tmp_path / "dedup.jsonl"
    raw = tmp_path / "raw.jsonl"
    met = tmp_path / "m.json"
    reg = tmp_path / "manifest.json"

    rows = [
        {"output_text": "a", "quality_score": 0.9, "quality_flags": []},
        {"output_text": "b", "quality_score": 0.7, "quality_flags": ["short_output"]},
    ]
    write_jsonl(dedup, rows)
    write_jsonl(raw, rows + [{"output_text": "dup"}])
    met.write_text(json.dumps({"exact_match": 0.5}), encoding="utf-8")

    r = subprocess.run(
        [
            sys.executable,
            "scripts/register_dataset.py",
            "--dataset-path",
            str(dedup),
            "--raw-dataset",
            str(raw),
            "--metrics",
            str(met),
            "--name",
            "papers_qa",
            "--version",
            "0.8.0",
            "--source-tag",
            "papers_qa",
            "--model",
            "gpt-4.1-mini",
            "--registry",
            str(reg),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    man = json.loads(reg.read_text(encoding="utf-8"))
    assert isinstance(man, list) and len(man) >= 1
    rec = man[-1]
    assert rec["name"] == "papers_qa" and rec["version"] == "0.8.0"
    assert rec["counts"]["rows"] == 2 and rec["counts"]["raw_rows"] == 3 and rec["counts"]["dedupe_dropped"] == 1
    assert "dataset_sha256" in rec["hashes"]
    assert rec["quality"]["rows_scored"] == 2
