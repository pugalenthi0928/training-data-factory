---
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

**Rows:** 298 (raw: 300, dedupe dropped: 2)
**Model used for eval:** `gpt-4.1-mini`
**Eval (n=50):**
- ROUGE-1 F: 0.724
- ROUGE-2 F: 0.566
- ROUGE-L F: 0.641
- Exact Match: 0.060

**Files**
- `papers_qa_only_real_gpt4_deduped.jsonl`
- `papers_qa_predictions_metrics.json`
- `manifest.json`

## Quick start (Python)

from datasets import load_dataset
ds = load_dataset(
    "json",
    data_files={"train": "hf://datasets/pugalenthi2000/tdf-papers-qa/papers_qa_only_real_gpt4_deduped.jsonl"},
    split="train",
)
print(len(ds), ds[0])
