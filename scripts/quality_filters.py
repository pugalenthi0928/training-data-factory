#!/usr/bin/env python3
import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from training_data_robo.io import load_jsonl, write_jsonl

REFUSAL_PATTERNS = [
    "as an ai language model","i cannot","i'm unable","i am unable","i won't","cannot assist","i do not have access",
]
STOPWORDS = set(["a", "an", "the", "and", "or", "but", "if", "while", "of", "for", "from", "to", "into", "onto", "in", "on", "at", "by", "with", "without", "within", "over", "under", "above", "below", "between", "among", "across", "after", "before", "during", "since", "until", "than", "then", "is", "are", "was", "were", "be", "being", "been", "am", "do", "does", "did", "doing", "have", "has", "had", "having", "can", "could", "may", "might", "must", "should", "would", "will", "you", "your", "yours", "they", "them", "their", "theirs", "we", "us", "our", "ours", "he", "she", "it", "its", "his", "her", "hers", "this", "that", "these", "those"])

def get_text(row, *cands):
    for c in cands:
        v = row.get(c)
        if isinstance(v, str) and v.strip():
            return v
    return ""

def tokens(text): return re.findall(r"[A-Za-z0-9']+", text.lower())

def bigram_repetition_ratio(text):
    toks = tokens(text)
    if len(toks) < 6: return 0.0
    bigrams = list(zip(toks, toks[1:]))
    if not bigrams: return 0.0
    cnt = Counter(bigrams)
    top = cnt.most_common(1)[0][1]
    return top / max(1, len(bigrams))

def has_refusal(text):
    t = text.lower()
    return any(pat in t for pat in REFUSAL_PATTERNS)

def content_words(text):
    return [t for t in tokens(text) if (t not in STOPWORDS and len(t) >= 4)]

def weak_grounding(answer, context, require_ratio=0.5):
    if not answer or not context: return False
    ans_terms = content_words(answer)
    if not ans_terms: return False
    ctx = context.lower()
    hits = sum(1 for t in ans_terms if t in ctx)
    needed = max(1, math.ceil(require_ratio * len(ans_terms)))
    return hits < needed

def min_len_for_task(task_name, default_min, min_summary, min_qa, min_keypoints, min_title):
    t = (task_name or "").lower()
    if "summary" in t: return min_summary
    if "qa" in t: return min_qa
    if "key_points" in t or "keypoints" in t or "key-points" in t: return min_keypoints
    if "title" in t: return min_title
    return default_min

def score_row(row, rep_thresh, min_summary, min_qa, min_keypoints, min_title):
    task = row.get("task_name", "")
    output = get_text(row, "output_text", "answer")
    question = get_text(row, "question")
    answer  = get_text(row, "answer", "output_text")
    context = get_text(row, "context", "input_text")
    flags = []

    needed = min_len_for_task(task, default_min=10, min_summary=min_summary, min_qa=min_qa,
                              min_keypoints=min_keypoints, min_title=min_title)
    if len(output.strip()) < needed: flags.append("short_output")
    if has_refusal(output): flags.append("possible_refusal")
    if bigram_repetition_ratio(output) >= rep_thresh: flags.append("high_repetition")
    if question and answer and context and weak_grounding(answer, context): flags.append("weak_grounding")

    score = 1.0
    if "short_output" in flags: score -= 0.40
    if "possible_refusal" in flags: score -= 0.60
    if "high_repetition" in flags: score -= 0.30
    if "weak_grounding" in flags: score -= 0.40
    row["quality_flags"] = flags
    row["quality_score"] = round(max(0.0, min(1.0, score)), 3)
    return row

def main():
    ap = argparse.ArgumentParser(description="Annotate rows with quality_flags + quality_score; optional dropping.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rep-threshold", type=float, default=0.45)
    ap.add_argument("--min-summary", type=int, default=80)
    ap.add_argument("--min-qa", type=int, default=20)
    ap.add_argument("--min-keypoints", type=int, default=40)
    ap.add_argument("--min-title", type=int, default=10)
    ap.add_argument("--drop-below", type=float, default=None)
    args = ap.parse_args()

    rows = load_jsonl(Path(args.input))
    out_rows = []
    for r in rows:
        r2 = score_row(r, args.rep_threshold, args.min_summary, args.min_qa, args.min_keypoints, args.min_title)
        if args.drop_below is None or r2["quality_score"] >= args.drop_below:
            out_rows.append(r2)

    write_jsonl(Path(args.output), out_rows)
    print(json.dumps({
        "input_rows": len(rows),
        "output_rows": len(out_rows),
        "dropped": len(rows) - len(out_rows),
        "rep_threshold": args.rep_threshold,
        "output": str(Path(args.output))
    }, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
