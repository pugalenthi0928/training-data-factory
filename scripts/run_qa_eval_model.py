from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
from typing import Any, Dict, List, Optional

def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows

def write_jsonl(p: Path, rows: List[Dict[str, Any]]) -> None:
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def norm(s: Optional[str]) -> str:
    return (s or "").strip()

def call_openai_chat(prompt: str, model: str, retries: int = 3, sleep_s: float = 1.5) -> str:
    # OpenAI Python SDK v1.x (compatible with gpt-4.1-mini via Chat)
    try:
        from openai import OpenAI  # type: ignore
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
    # Deterministic, test-friendly prediction.
    # If answer tokens appear in context, return answer; else "Unknown".
    q = norm(question).lower()
    a = norm(answer)
    c = norm(context).lower()
    if a and a.lower() in c:
        return a
    # fallbacks: try to pick a short plausible fragment from context
    return "Unknown"

def build_prompt(question: str, context: str) -> str:
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nAnswer:"

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate QA predictions for evaluation.")
    ap.add_argument("--input", required=True, help="Input JSONL with question/answer/context rows.")
    ap.add_argument("--output", required=True, help="Output JSONL with added 'prediction'.")
    ap.add_argument("--model", default="gpt-4.1-mini", help="OpenAI chat model (when not using --fake-model).")
    ap.add_argument("--max-examples", type=int, default=None, help="Optional cap on examples.")
    ap.add_argument("--fake-model", action="store_true", help="Use offline fake model (no API calls).")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    rows = load_jsonl(in_path)
    if not rows:
        raise SystemExit("Input dataset is empty; nothing to predict.")

    # Keep only rows with QA signals
    filt: List[Dict[str, Any]] = []
    for r in rows:
        if ("question" in r and "answer" in r) or str(r.get("task_type","")).lower() == "qa" or "qa" in str(r.get("task_name","")).lower():
            filt.append(r)
    if not filt:
        raise SystemExit("No QA-like rows found (need question/answer or task_name contains 'qa').")

    if args.max_examples is not None:
        filt = filt[: args.max_examples]

    out_rows: List[Dict[str, Any]] = []
    for r in filt:
        question = norm(r.get("question"))
        context = norm(r.get("context"))
        answer = norm(r.get("answer"))
        if not question:
            # Skip rows without a question
            continue

        if args.fake-model:
            pred = fake_predict(question, context, answer)
        else:
            prompt = build_prompt(question, context)
            pred = call_openai_chat(prompt, model=args.model)

        new_r = dict(r)
        new_r["prediction"] = pred
        out_rows.append(new_r)

    if not out_rows:
        raise SystemExit("No rows produced (check filters).")

    write_jsonl(out_path, out_rows)
    summary = {
        "input_rows": len(rows),
        "predicted_rows": len(out_rows),
        "model": ("fake" if args.fake-model else args.model),
        "output_path": str(out_path),
    }
    print("Prediction summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
