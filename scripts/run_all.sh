#!/usr/bin/env bash
set -e

echo "=== Running tdr profiles ==="
python scripts/run_profile.py --profile small_sample
python scripts/run_profile.py --profile qa_only

echo
echo "=== Running quality post-processing ==="
if [ -f output/dataset_cli_rich_200.jsonl ]; then
  python scripts/postprocess_quality.py \
    --input output/dataset_cli_rich_200.jsonl \
    --output output/dataset_cli_rich_200_quality.jsonl
fi
if [ -f output/papers_qa_only_real_gpt4.jsonl ]; then
  python scripts/postprocess_quality.py \
    --input output/papers_qa_only_real_gpt4.jsonl \
    --output output/papers_qa_only_real_gpt4_quality.jsonl
fi

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
echo "=== Exporting fine-tune and RAG datasets ==="
if [ -f output/dataset_cli_rich_200.jsonl ]; then
  python scripts/export_finetune.py \
    --input output/dataset_cli_rich_200.jsonl \
    --output output/finetune_summary_text.jsonl \
    --task-name summary_v1 \
    --format text
fi

if [ -f output/papers_qa_only_real_gpt4.jsonl ]; then
  python scripts/export_finetune.py \
    --input output/papers_qa_only_real_gpt4.jsonl \
    --output output/finetune_qa_text.jsonl \
    --task-name qa_v1 \
    --format text

  python scripts/export_rag_qa.py \
    --input output/papers_qa_only_real_gpt4.jsonl \
    --output output/papers_rag_qa.jsonl \
    --task-name qa_v1
fi

echo
echo "=== Comparing datasets ==="
if [ -f scripts/compare_datasets.py ]; then
  if [ -f output/dataset_cli_rich_200.jsonl ] && [ -f output/papers_qa_only_real_gpt4.jsonl ]; then
    python scripts/compare_datasets.py \
      --inputs output/dataset_cli_rich_200.jsonl output/papers_qa_only_real_gpt4.jsonl \
      --output output/dataset_comparison.json
  fi
fi

echo
echo "All done 🎉"
