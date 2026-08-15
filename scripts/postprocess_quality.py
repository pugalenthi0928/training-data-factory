from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from training_data_robo.io import load_jsonl, write_jsonl

REFUSAL_PATTERNS: List[str] = [
    "as an ai language model",
    "i am an ai language model",
    "i'm an ai language model",
    "i cannot",
    "i can't",
    "i am unable to",
    "i'm unable to",
    "cannot provide",
    "cannot answer",
    "sorry, but i",
]


def normalise_row(row: Dict[str, Any]) -> Dict[str, Any]:
    if "input_text" not in row:
        for fallback in ("input", "context", "question"):
            if fallback in row:
                row["input_text"] = str(row[fallback])
                break
        else:
            row["input_text"] = ""
    if "output_text" not in row:
        for fallback in ("output", "answer"):
            if fallback in row:
                row["output_text"] = str(row[fallback])
                break
        else:
            row["output_text"] = ""
    return row


def score_example(row: Dict[str, Any]) -> Tuple[List[str], float]:
    flags: List[str] = []

    output = str(row.get("output_text", "") or "")
    answer = str(row.get("answer", "") or "")
    context = str(row.get("context", "") or "")
    task_name = str(row.get("task_name", "") or "").lower()
    task_type = str(row.get("task_type", "") or "").lower()

    score = 1.0

    if not output.strip():
        flags.append("empty_output")
        score -= 0.5

    out_len = len(output)
    min_len = 0
    if "summary" in task_name:
        min_len = 80
    elif "key_points" in task_name or "keypoints" in task_name:
        min_len = 60
    elif "title" in task_name or task_type == "qa" or "qa" in task_name:
        min_len = 10
    if min_len and out_len < min_len:
        flags.append("short_output")
        score -= 0.2

    low = output.lower()
    if low:
        for pat in REFUSAL_PATTERNS:
            if pat in low:
                flags.append("possible_refusal")
                score -= 0.3
                break

    tokens = output.split()
    if tokens:
        counts = Counter(tokens)
        most_common = counts.most_common(1)[0][1]
        if most_common / max(1, len(tokens)) > 0.5 and len(tokens) >= 20:
            flags.append("repetitive_output")
            score -= 0.2

    if answer.strip() and context.strip():
        answer_tokens = set(answer.lower().split())
        context_tokens = set(context.lower().split())
        if answer_tokens:
            overlap = len(answer_tokens & context_tokens)
            if overlap < max(1, int(0.2 * len(answer_tokens))):
                flags.append("weak_grounding")
                score -= 0.2

    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0

    flags = sorted(set(flags))
    return flags, float(score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add quality_flags and quality_score to a JSONL dataset.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL dataset with quality annotations.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    records = load_jsonl(in_path)
    if not records:
        raise SystemExit("Dataset is empty; nothing to process.")

    flag_counter: Counter[str] = Counter()
    num_rows = len(records)
    processed: List[Dict[str, Any]] = []

    for row in records:
        row = normalise_row(row)
        flags, score = score_example(row)
        row["quality_flags"] = flags
        row["quality_score"] = score
        for f in flags:
            flag_counter[f] += 1
        processed.append(row)

    write_jsonl(out_path, processed)

    summary = {
        "num_examples": num_rows,
        "flag_counts": dict(flag_counter),
        "input": str(in_path),
        "output": str(out_path),
    }
    print("Quality post-processing summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
