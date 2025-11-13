#!/usr/bin/env python3
import json, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--manifest", default="registry/manifest.json")
    ap.add_argument("--metrics", default="output/papers_qa_predictions_metrics.json")
    ap.add_argument("--out", default="README.md")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rec = manifest[-1] if isinstance(manifest, list) and manifest else manifest
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))

    readme = f"""---
dataset_info:
  features:
  - name: input_text
    dtype: string
  - name: output_text
    dtype: string
  - name: quality_score
    dtype: float64
license: other
language:
- en
task_categories:
- question-answering
pretty_name: Papers QA (generated)
size_categories:
- 1K<n<10K
---

# Papers QA (deduped)

**Rows:** {rec['counts']['rows']} (raw: {rec['counts']['raw_rows']}, dedupe dropped: {rec['counts']['dedupe_dropped']})
**Model used for eval:** `{rec['model']}`
**Eval (n={metrics['num_eval_examples']}):**
- ROUGE-1 F: {metrics['rouge1_f']:.3f}
- ROUGE-2 F: {metrics['rouge2_f']:.3f}
- ROUGE-L F: {metrics['rougeL_f']:.3f}
- Exact Match: {metrics['exact_match']:.3f}

**Files**
- `papers_qa_only_real_gpt4_deduped.jsonl`
- `papers_qa_predictions_metrics.json`
- `manifest.json`

## Quick start (Python)

from datasets import load_dataset
ds = load_dataset(
    "json",
    data_files={{"train": "hf://datasets/{args.repo}/papers_qa_only_real_gpt4_deduped.jsonl"}},
    split="train",
)
print(len(ds), ds[0])
"""
    Path(args.out).write_text(readme, encoding="utf-8")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
