from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from training_data_robo.io import load_jsonl, write_jsonl


def norm(s: Optional[str]) -> str:
    return (s or "").strip()

def call_openai_chat(prompt: str, model: str, retries: int = 3, sleep_s: float = 1.5) -> str:
    try:
        from openai import OpenAI  # OpenAI SDK v1.x
    except Exception as e:
        raise RuntimeError("OpenAI SDK not installed. `pip install openai`") from e

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set in env")

    client = OpenAI()
    last_err: Optional[Exception] = None
    for _ in range(retries):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Answer concisely using ONLY the provided CONTEXT. If the answer is not in CONTEXT, reply exactly 'Unknown'."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    raise RuntimeError(f"OpenAI call failed after {retries} retries: {last_err}")

def fake_predict(question: str, context: str, answer: str) -> str:
    # Deterministic, test-friendly predictor:
    a = norm(answer)
    c = norm(context).lower()
    if a and a.lower() in c:
        return a
    return "Unknown"

def build_prompt(question: str, context: str) -> str:
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer:"

def extract_question(row: Dict[str, Any]) -> str:
    # Prefer explicit question
    q = norm(row.get("question"))
    if q:
        return q
    # Common fallbacks used by various templates
    meta = row.get("metadata") or {}
    if isinstance(meta, dict):
        q = norm(meta.get("question"))
        if q:
            return q
    for k in ("input_question", "prompt", "query"):
        q = norm(row.get(k))
        if q:
            return q
    # If it's a QA task_name but author only stored input_text as the question
    if "qa" in str(row.get("task_name","")).lower() and not norm(row.get("context")):
        q = norm(row.get("input_text"))
        if q:
            return q
    return ""

def extract_context(row: Dict[str, Any]) -> str:
    c = norm(row.get("context"))
    if c:
        return c
    # Fall back to input_text for RAG-like rows
    return norm(row.get("input_text"))

def extract_answer(row: Dict[str, Any]) -> str:
    a = norm(row.get("answer"))
    if a:
        return a
    return norm(row.get("output_text"))

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate QA predictions for evaluation.")
    ap.add_argument("--input", required=True, help="Input JSONL with QA-like rows.")
    ap.add_argument("--output", required=True, help="Output JSONL with added 'prediction'.")
    ap.add_argument("--model", default="gpt-4.1-mini", help="OpenAI chat model when not using --fake-model.")
    ap.add_argument("--max-examples", type=int, default=None, help="Optional cap on examples.")
    ap.add_argument("--fake-model", dest="fake_model", action="store_true", help="Use offline fake model (no API calls).")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    rows = load_jsonl(in_path)
    if not rows:
        raise SystemExit("Input dataset is empty; nothing to predict.")

    # Keep rows that look QA-ish
    qaish: List[Dict[str, Any]] = []
    for r in rows:
        tn = str(r.get("task_name","")).lower()
        tt = str(r.get("task_type","")).lower()
        q = norm(r.get("question"))
        has_qa_signal = ("qa" in tn) or ("qa" in tt) or bool(q)
        if has_qa_signal:
            qaish.append(r)
    if not qaish:
        raise SystemExit("No QA-like rows found (need question/task_name or task_type including 'qa').")

    if args.max_examples is not None:
        qaish = qaish[: args.max_examples]

    out_rows: List[Dict[str, Any]] = []
    for r in qaish:
        question = extract_question(r)
        context  = extract_context(r)
        answer   = extract_answer(r)

        if not question:
            # Still no question → skip this row
            continue

        if args.fake_model:
            pred = fake_predict(question, context, answer)
        else:
            pred = call_openai_chat(build_prompt(question, context), model=args.model)

        new_r = dict(r)
        new_r["prediction"] = pred
        out_rows.append(new_r)

    if not out_rows:
        raise SystemExit("No rows produced (check filters / question field).")

    write_jsonl(out_path, out_rows)
    summary = {
        "input_rows": len(rows),
        "predicted_rows": len(out_rows),
        "model": ("fake" if args.fake_model else args.model),
        "output_path": str(out_path),
    }
    print("Prediction summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
