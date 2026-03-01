"""Diversity scoring for training datasets.

Measures how diverse a set of training examples are using text-based
heuristics (no API calls required) and optionally embedding-based metrics.

Metrics:
  - Vocabulary diversity: unique tokens / total tokens
  - Length variance: coefficient of variation of output lengths
  - Task coverage: number of unique task types represented
  - Redundancy ratio: fraction of near-duplicate pairs (via Jaccard similarity)
"""
from __future__ import annotations

import string
from collections import Counter
from typing import Any, Dict, List, Set

from .logging_config import get_logger

logger = get_logger("training_data_robo.diversity")

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _tokenize_simple(text: str) -> List[str]:
    return text.lower().translate(_PUNCT_TABLE).split()


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def compute_diversity_metrics(
    examples: List[Dict[str, Any]],
    text_field: str = "output_text",
    jaccard_sample_size: int = 200,
    jaccard_threshold: float = 0.7,
) -> Dict[str, Any]:
    """Compute diversity metrics for a list of examples.

    Args:
        examples: List of dicts with at least `text_field`.
        text_field: Which field to analyze.
        jaccard_sample_size: Max examples to use for pairwise Jaccard (O(n^2)).
        jaccard_threshold: Jaccard similarity above this = "redundant" pair.

    Returns:
        Dict with diversity metrics.
    """
    if not examples:
        return {
            "num_examples": 0,
            "vocab_diversity": 0.0,
            "length_cv": 0.0,
            "task_coverage": 0,
            "unique_tasks": [],
            "redundancy_ratio": 0.0,
            "avg_output_length": 0.0,
        }

    texts = [str(ex.get(text_field, "")) for ex in examples]
    all_tokens: List[str] = []
    lengths: List[int] = []
    token_sets: List[Set[str]] = []

    for t in texts:
        toks = _tokenize_simple(t)
        all_tokens.extend(toks)
        lengths.append(len(t))
        token_sets.append(set(toks))

    # Vocabulary diversity
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    vocab_diversity = unique_tokens / max(1, total_tokens)

    # Length coefficient of variation
    avg_len = sum(lengths) / max(1, len(lengths))
    if avg_len > 0:
        variance = sum((ln - avg_len) ** 2 for ln in lengths) / max(1, len(lengths))
        std_dev = variance ** 0.5
        length_cv = std_dev / avg_len
    else:
        length_cv = 0.0

    # Task coverage
    tasks = [str(ex.get("task_name", "unknown")) for ex in examples]
    unique_tasks = sorted(set(tasks))
    task_counts = Counter(tasks)

    # Redundancy ratio (pairwise Jaccard on a sample)
    sample = token_sets[:jaccard_sample_size]
    redundant_pairs = 0
    total_pairs = 0
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            total_pairs += 1
            if _jaccard(sample[i], sample[j]) >= jaccard_threshold:
                redundant_pairs += 1

    redundancy_ratio = redundant_pairs / max(1, total_pairs)

    return {
        "num_examples": len(examples),
        "vocab_diversity": round(vocab_diversity, 4),
        "length_cv": round(length_cv, 4),
        "task_coverage": len(unique_tasks),
        "unique_tasks": unique_tasks,
        "task_distribution": dict(task_counts),
        "redundancy_ratio": round(redundancy_ratio, 4),
        "avg_output_length": round(avg_len, 1),
        "total_unique_tokens": unique_tokens,
        "total_tokens": total_tokens,
    }
