import json
import subprocess
import sys
from pathlib import Path


def wjsonl(p: Path, rows):
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def rjsonl(p: Path):
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def test_quality_flags_and_score(tmp_path: Path):
    inp = tmp_path / "data.jsonl"
    outp = tmp_path / "scored.jsonl"
    rows = [
        {"task_name":"qa_v1","question":"Capital of France?","answer":"Paris is the capital of France.","context":"France's capital city is Paris.","output_text":"Paris is the capital of France."},
        {"task_name":"summary_v1","output_text":"Too short"},
        {"task_name":"qa_v1","question":"Tell me X","answer":"N/A","context":"N/A","output_text":"As an AI language model, I cannot provide that."},
        {"task_name":"key_points_v1","output_text":"alpha beta alpha beta alpha beta alpha beta"},
        {"task_name":"qa_v1","question":"Where is Eiffel Tower?","answer":"The Eiffel Tower is in Paris.","context":"This passage mentions famous museums in Paris.","output_text":"The Eiffel Tower is in Paris."}
    ]
    wjsonl(inp, rows)
    r = subprocess.run([sys.executable, "scripts/quality_filters.py", "--input", str(inp), "--output", str(outp)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    scored = rjsonl(outp)
    flags = [set(x.get("quality_flags", [])) for x in scored]
    assert "weak_grounding" not in flags[0]
    assert "short_output" in flags[1]
    assert "possible_refusal" in flags[2]
    assert "high_repetition" in flags[3]
    assert "weak_grounding" in flags[4]
    for row in scored:
        s = row.get("quality_score")
        assert isinstance(s, float) and 0.0 <= s <= 1.0
