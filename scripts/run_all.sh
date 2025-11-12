#!/usr/bin/env bash
set -euo pipefail

echo "=== Running tdr profiles ==="
python scripts/run_profile.py --profile small_sample || true
python scripts/run_profile.py --profile qa_only || true

echo
echo "=== Running quality post-processing ==="
if [ -f scripts/postprocess_quality.py ] && [ -f output/dataset_cli_rich_200.jsonl ]; then
  python scripts/postprocess_quality.py \
    --input output/dataset_cli_rich_200.jsonl \
    --output output/dataset_cli_rich_200_quality.jsonl
fi
if [ -f scripts/postprocess_quality.py ] && [ -f output/papers_qa_only_real_gpt4.jsonl ]; then
  python scripts/postprocess_quality.py \
    --input output/papers_qa_only_real_gpt4.jsonl \
    --output output/papers_qa_only_real_gpt4_quality.jsonl
fi

echo
echo "=== Generating predictions for evaluation (papers QA) ==="
if [ -f output/papers_qa_only_real_gpt4.jsonl ]; then
  python scripts/run_qa_eval_model.py \
    --input output/papers_qa_only_real_gpt4.jsonl \
    --output output/papers_qa_predictions.jsonl \
    --model gpt-4.1-mini \
    --max-examples 50
fi

echo
echo "=== Running QA evaluation ==="
if [ -f output/papers_qa_predictions.jsonl ]; then
  python scripts/evaluate_qa.py \
    --input output/papers_qa_predictions.jsonl \
    --output output/papers_qa_predictions_metrics.json
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
