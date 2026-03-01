from __future__ import annotations

import argparse
import json
import os
import string
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from training_data_robo.io import load_jsonl, write_jsonl

_PUNCT = str.maketrans("", "", string.punctuation)

def bow_signature(text: str) -> str:
    """
    Bag-of-words signature:
      - lowercase
      - strip punctuation
      - split on whitespace
      - sort tokens (order-invariant)
    """
    toks = text.lower().translate(_PUNCT).split()
    toks.sort()
    return " ".join(toks)

def ensure_flags(row: Dict[str, Any]) -> None:
    if "quality_flags" not in row or row["quality_flags"] in ("", None):
        row["quality_flags"] = []
    if not isinstance(row["quality_flags"], list):
        try:
            row["quality_flags"] = list(row["quality_flags"])
        except Exception:
            row["quality_flags"] = [str(row["quality_flags"])]

def dedupe_hash(rows: List[Dict[str, Any]], text_field: str) -> List[Dict[str, Any]]:
    seen = set()
    kept: List[Dict[str, Any]] = []
    for r in rows:
        key = bow_signature(str(r.get(text_field, "")))
        if key in seen:
            ensure_flags(r)
            r["quality_flags"].append("near_duplicate")
            continue
        seen.add(key)
        kept.append(r)
    return kept

def get_embeddings_openai(texts: List[str], model: str):
    # Lazy import so CI without OpenAI still passes
    import numpy as np
    from openai import OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing")
    client = OpenAI()
    vecs = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        res = client.embeddings.create(model=model, input=batch)
        vecs.extend([d.embedding for d in res.data])
    arr = np.array(vecs, dtype="float32")
    norms = (arr**2).sum(1, keepdims=True) ** 0.5 + 1e-12
    return arr / norms

def dedupe_emb(rows: List[Dict[str, Any]], text_field: str, model: str, threshold: float) -> List[Dict[str, Any]]:
    import numpy as np
    texts = [str(r.get(text_field, "")) for r in rows]
    if not texts:
        return rows
    vecs = get_embeddings_openai(texts, model=model)
    kept_idx: List[int] = []
    dup = set()
    for i in range(len(rows)):
        if i in dup:
            continue
        keep = True
        for k in kept_idx:
            sim = float(np.dot(vecs[i], vecs[k]))
            if sim >= threshold:
                dup.add(i)
                ensure_flags(rows[i])
                rows[i]["quality_flags"].append("near_duplicate")
                keep = False
                break
        if keep:
            kept_idx.append(i)
    return [rows[i] for i in range(len(rows)) if i not in dup]

def main():
    ap = argparse.ArgumentParser(description="Near-duplicate removal (embeddings or hash).")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--text-field", default="output_text")
    ap.add_argument("--method", choices=["emb","hash"], default="emb")
    ap.add_argument("--threshold", type=float, default=0.92)
    ap.add_argument("--emb-model", default="text-embedding-3-small")
    ap.add_argument("--max-examples", type=int, default=None)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    rows = load_jsonl(in_path)
    if not rows:
        raise SystemExit("Input is empty.")
    if args.max_examples:
        rows = rows[: args.max_examples]

    if args.method == "hash":
        kept = dedupe_hash(rows, args.text_field)
        write_jsonl(out_path, kept)
        print(json.dumps({
            "method": "hash_bow",
            "input_rows": len(rows),
            "kept_rows": len(kept),
            "dropped_duplicates": len(rows) - len(kept),
            "text_field": args.text_field,
            "output_path": str(out_path),
        }, indent=2, ensure_ascii=False))
        return

    try:
        kept = dedupe_emb(rows, args.text_field, args.emb_model, args.threshold)
        method_used = "emb"
        dropped = len(rows) - len(kept)
    except Exception as e:
        kept = dedupe_hash(rows, args.text_field)
        method_used = "hash_fallback_bow"
        dropped = len(rows) - len(kept)
        print(f"[compute_dedupe] Embedding path failed ({e}); used hash fallback.", flush=True)

    write_jsonl(out_path, kept)
    print(json.dumps({
        "method": method_used,
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "dropped_duplicates": dropped,
        "threshold": args.threshold,
        "emb_model": args.emb_model,
        "text_field": args.text_field,
        "output_path": str(out_path),
    }, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
