# Forge - Training Data Engine That Proves Its Own Worth

[![CI](https://github.com/pugalenthi0928/training-data-factory/actions/workflows/ci.yml/badge.svg)](https://github.com/pugalenthi0928/training-data-factory/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Forge does not just generate training data. It proves the data works.**

Generate multi-strategy training data from documents, evaluate it with LLM-as-judge rubrics, detect benchmark contamination, fine-tune a model locally on Apple Silicon, and benchmark before/after with statistical significance testing. One command, end to end.

**[Demo](https://pugalenthi0928.github.io/training-data-factory/) | [Overview](https://pugalenthi0928.github.io/training-data-factory/overview.html) | [Technical Deep-Dive](https://pugalenthi0928.github.io/training-data-factory/technical.html)**

## Architecture

```mermaid
graph LR
    A[Documents] --> B[Chunk]
    B --> C[Generate]
    C --> D[Quality Score]
    D --> E[Deduplicate]
    E --> F[LLM-as-Judge]
    F --> G[Contamination Check]
    G --> H[Difficulty Calibrate]
    H --> I[Select + Split]
    I --> J[Fine-Tune MLX]
    J --> K[Benchmark]
```

## Quick Start

```bash
# Install
git clone https://github.com/pugalenthi0928/training-data-factory.git
cd training-data-factory
make install

# Dry run (no API calls, uses DummyLLM)
make forge

# Full pipeline with real models
export OPENAI_API_KEY=sk-...
make forge-live

# Dashboard
make dashboard
```

## What Makes This Different

| Feature | Most Projects | Forge |
|---------|--------------|-------|
| Data generation | Single task (QA only) | 6 task types: QA, summary, key points, title, instruction-following, chain-of-thought |
| Quality evaluation | Manual spot-check | LLM-as-judge with 4-dimension rubric (faithfulness, helpfulness, complexity, coherence) |
| Contamination | Ignored | N-gram overlap detection against benchmark datasets |
| Proof it works | "Trust me" | Fine-tune + benchmark with statistical significance testing |
| Pipeline | Bash scripts | DAG runner with caching, resume, and experiment tracking |
| Chunking | Naive paragraph split | Structure-aware: detects headers, lists, tables |

## Pipeline Steps

1. **Generate** - Multi-strategy training data from source documents via GPT-4.1-mini
2. **Quality Score** - Heuristic quality flags (length, refusal detection, readability)
3. **Deduplicate** - Hash-based and embedding-based deduplication
4. **LLM-as-Judge** - GPT-4.1-mini scores each example on 4 rubric dimensions (1-5 scale)
5. **Contamination Check** - 8-gram and 13-gram overlap detection against benchmark datasets
6. **Difficulty Calibration** - Heuristic difficulty tagging (easy/medium/hard)
7. **Select** - Top-N selection via quality-weighted, diverse, balanced, or curriculum strategies
8. **Train/Test Split** - Stratified split preserving task-type proportions
9. **Fine-Tune** - LoRA fine-tuning on Apple Silicon via MLX (Qwen 2.5 0.5B, zero GPU cost)
10. **Benchmark** - Base vs fine-tuned comparison with ROUGE, exact match, and paired bootstrap significance

## Project Structure

```
src/training_data_robo/
├── bot.py              # High-level orchestrator
├── cli.py              # CLI entry point (tdr process)
├── models.py           # Domain models (TaskType, TrainingExample, ...)
├── chunking.py         # Structure-aware document chunking
├── task_selector.py    # Adaptive task selection per chunk type
├── quality.py          # Quality filtering (refusals, length, dedup)
├── judge.py            # LLM-as-judge with rubric evaluation
├── contamination.py    # N-gram contamination detection
├── diversity.py        # Vocabulary & task diversity metrics
├── difficulty.py       # Difficulty calibration (easy/medium/hard)
├── selector.py         # Data selection strategies
├── pipeline.py         # DAG pipeline runner with caching & resume
├── tracker.py          # Lightweight experiment tracker
├── io.py               # Consolidated JSONL I/O
├── ai_client.py        # LLM client abstraction (OpenAI + Dummy)
└── settings.py         # Configuration

scripts/
├── run_forge.py        # One-command pipeline orchestrator
├── run_judge.py        # LLM-as-judge CLI
├── check_contamination.py  # Contamination detection CLI
├── calibrate_difficulty.py # Difficulty calibration CLI
├── split_dataset.py    # Stratified train/test split
├── finetune_mlx.py     # MLX LoRA fine-tuning
├── benchmark.py        # Base vs fine-tuned benchmarking
└── postprocess_quality.py  # Quality scoring CLI

app.py                  # Streamlit dashboard (5 pages)
```

## Dashboard

The Streamlit dashboard (`make dashboard`) provides:

- **Pipeline Overview** - Step status, timing, output files per run
- **Quality Deep-Dive** - Judge score distributions, difficulty breakdown, contamination results
- **Training Results** - Base vs fine-tuned comparison with delta charts and significance
- **Experiment Comparison** - Multi-run metric comparison table
- **Dataset Explorer** - Browse, filter, search, and curate individual examples

## Development

```bash
make lint        # Ruff linting
make format      # Auto-format
make typecheck   # mypy
make test        # pytest
make coverage    # pytest with coverage report
```

## Results

From a real pipeline run on ML/AI technical documents (200 examples, 4 task types):

**Data Quality (LLM-as-Judge, GPT-4.1-mini):**
- Average score: **4.77/5.0** across 4 dimensions (faithfulness, helpfulness, complexity, coherence)
- 86% scored 5/5, 14% scored 4/5, 0% below 4/5
- Only 2 quality flags (short_output) out of 200 examples
- 0 duplicates detected

**Fine-Tuning (LoRA on Qwen 2.5 0.5B, Apple Silicon MLX):**

| Metric | Base Model | Fine-Tuned | Delta |
|--------|-----------|------------|-------|
| ROUGE-1 | 0.418 | 0.586 | **+16.8%** |
| ROUGE-2 | 0.185 | 0.311 | **+12.7%** |
| ROUGE-L | 0.289 | 0.417 | **+12.9%** |
| Statistical significance | — | — | **p=0.0** |

Training: 120 iterations, 5 minutes, val loss 2.345 → 0.730 (69% reduction). 0.594% of parameters trained (2.9M/494M).

## Cost

| Component | Cost |
|-----------|------|
| Generation (200 examples, 4 task types) | ~$3-8 |
| LLM-as-Judge (200 examples x 4 dimensions) | ~$1-2 |
| Embedding dedup + diversity | ~$0.02 |
| MLX fine-tuning (local Apple Silicon) | $0 |
| **Total per run** | **~$5-10** |

## Limitations

- **Fine-tuning** requires Apple Silicon (M1+) for MLX. Falls back to OpenAI fine-tuning API otherwise.
- **Contamination detection** uses n-gram overlap. It will not catch paraphrased benchmark leakage.
- **LLM-as-judge** scores correlate with but do not replace human evaluation.
- **Small model fine-tuning** (0.5B) shows proof-of-concept improvement. Production would use larger models.
- **Generation quality** depends heavily on source document quality and chunk boundaries.

## License

MIT
