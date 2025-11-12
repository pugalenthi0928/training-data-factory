#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root
cd "$(dirname "$0")/.."

echo "=== Running tdr profiles ==="
python scripts/run_profile.py --profile small_sample
python scripts/run_profile.py --profile qa_only

echo
echo "=== Running QA evaluation ==="
if [ -f output/papers_qa_only_real_gpt4.jsonl ]; then
  python scripts/evaluate_qa.py \
    --input output/papers_qa_only_real_gpt4.jsonl \
    --output output/papers_qa_only_real_gpt4_metrics.json
fi

if [ -f output/dataset_cli_rich_200.jsonl ]; then
  python scripts/evaluate_qa.py \
    --input output/dataset_cli_rich_200.jsonl \
    --output output/dataset_cli_rich_200_metrics.json
fi

echo
echo "=== Generating dataset cards ==="
if [ -f scripts/generate_dataset_card.py ]; then
  if [ -f output/dataset_cli_rich_200.jsonl ]; then
    python scripts/generate_dataset_card.py \
      --input output/dataset_cli_rich_200.jsonl \
      --output output/dataset_cli_rich_200_card.md
  fi
  if [ -f output/papers_qa_only_real_gpt4.jsonl ]; then
    python scripts/generate_dataset_card.py \
      --input output/papers_qa_only_real_gpt4.jsonl \
      --output output/papers_qa_only_real_gpt4_card.md
  fi
fi

echo
echo "All done 🎉"
