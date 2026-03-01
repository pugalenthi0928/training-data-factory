#!/usr/bin/env bash
set -euo pipefail

echo "=== Running tdr profiles ==="
tdr process --source ./sample_docs --out output/dataset_cli_rich_200.jsonl --tasks summary,qa,key_points,title --max-examples 200 --max-chars 900 --overlap 120 --model gpt-4.1-mini
tdr process --source ../Papers --out output/papers_qa_only_real_gpt4.jsonl --tasks qa --max-examples 300 --max-chars 900 --overlap 120 --model gpt-4.1-mini

echo
echo "=== Running quality post-processing (basic) ==="
python scripts/postprocess_quality.py --input output/dataset_cli_rich_200.jsonl --output output/dataset_cli_rich_200_quality.jsonl
python scripts/postprocess_quality.py --input output/papers_qa_only_real_gpt4.jsonl --output output/papers_qa_only_real_gpt4_quality.jsonl

echo
echo "=== Scoring & flagging quality ==="
python scripts/quality_filters.py --input output/dataset_cli_rich_200_quality.jsonl --output output/dataset_cli_rich_200_quality_scored.jsonl
python scripts/quality_filters.py --input output/papers_qa_only_real_gpt4_quality.jsonl --output output/papers_qa_only_real_gpt4_quality_scored.jsonl

echo
echo "=== Dedupe (embeddings with hash fallback) ==="
python scripts/compute_dedupe.py --input output/dataset_cli_rich_200_quality_scored.jsonl --output output/dataset_cli_rich_200_deduped.jsonl --text-field output_text --method emb --threshold 0.92 --emb-model text-embedding-3-small || python scripts/compute_dedupe.py --input output/dataset_cli_rich_200_quality_scored.jsonl --output output/dataset_cli_rich_200_deduped.jsonl --method hash
python scripts/compute_dedupe.py --input output/papers_qa_only_real_gpt4_quality_scored.jsonl --output output/papers_qa_only_real_gpt4_deduped.jsonl --text-field output_text --method emb --threshold 0.92 --emb-model text-embedding-3-small || python scripts/compute_dedupe.py --input output/papers_qa_only_real_gpt4_quality_scored.jsonl --output output/papers_qa_only_real_gpt4_deduped.jsonl --method hash

echo
echo "=== Generating predictions for evaluation (papers QA; using deduped) ==="
python scripts/run_qa_eval_model.py \
  --input output/papers_qa_only_real_gpt4_deduped.jsonl \
  --output output/papers_qa_predictions.jsonl \
  --model gpt-4.1-mini \
  --max-examples 50

echo
echo "=== Running QA evaluation ==="
python scripts/evaluate_qa.py \
  --input output/papers_qa_predictions.jsonl \
  --output output/papers_qa_predictions_metrics.json \
  --reference-column output_text \
  --prediction-column prediction

echo
echo "=== Logging leaderboard row ==="
python scripts/log_metrics.py \
  --metrics output/papers_qa_predictions_metrics.json \
  --dataset output/papers_qa_only_real_gpt4_deduped.jsonl \
  --model gpt-4.1-mini

echo
echo "=== Generating dataset cards ==="
if [ -f scripts/generate_dataset_card.py ]; then
  if [ -f output/dataset_cli_rich_200.jsonl ]; then
    python scripts/generate_dataset_card.py --input output/dataset_cli_rich_200.jsonl --output output/dataset_cli_rich_200_card.md
  fi
  if [ -f output/papers_qa_only_real_gpt4.jsonl ]; then
    python scripts/generate_dataset_card.py --input output/papers_qa_only_real_gpt4.jsonl --output output/papers_qa_only_real_gpt4_card.md
  fi
fi

echo
echo "All done 🎉"

echo
echo "=== Register dataset ==="
DATASET_VERSION="${DATASET_VERSION:-0.8.0}"
python scripts/register_dataset.py \
  --dataset-path output/papers_qa_only_real_gpt4_deduped.jsonl \
  --raw-dataset output/papers_qa_only_real_gpt4.jsonl \
  --metrics output/papers_qa_predictions_metrics.json \
  --name papers_qa \
  --version "${DATASET_VERSION}" \
  --source-tag papers_qa \
  --model gpt-4.1-mini \
  --registry registry/manifest.json

if [ "${PUBLISH_HF:-0}" = "1" ]; then
  echo
  echo "=== Publish to Hugging Face ==="
  python scripts/publish_hf.py \
    --repo "${HF_REPO:?Set HF_REPO, e.g. yourname/tdf-papers-qa}" \
    --path output/papers_qa_only_real_gpt4_deduped.jsonl \
    --path output/papers_qa_predictions_metrics.json \
    --path registry/manifest.json
fi
