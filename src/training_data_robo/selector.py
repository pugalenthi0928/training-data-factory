"""Data selection strategies for choosing the best training examples.

Provides multiple strategies for selecting a subset of examples
for fine-tuning, balancing quality, diversity, and curriculum design.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from .logging_config import get_logger

logger = get_logger("training_data_robo.selector")


def select_quality_weighted(
    examples: List[Dict[str, Any]],
    n: int,
    score_field: str = "judge_avg_score",
    fallback_field: str = "quality_score",
) -> List[Dict[str, Any]]:
    """Select top-N examples by quality score (highest first)."""
    def _score(ex: Dict[str, Any]) -> float:
        s = ex.get(score_field)
        if s is not None:
            return float(s)
        s = ex.get(fallback_field)
        if s is not None:
            return float(s)
        return 0.0

    ranked = sorted(examples, key=_score, reverse=True)
    return ranked[:n]


def select_diverse(
    examples: List[Dict[str, Any]],
    n: int,
    text_field: str = "output_text",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Select N examples maximizing token-level diversity via greedy coverage.

    Iteratively picks the example whose token set has the least overlap
    with already-selected examples.
    """
    if n >= len(examples):
        return list(examples)

    rng = random.Random(seed)

    # Tokenize all examples
    token_sets = []
    for ex in examples:
        tokens = set(str(ex.get(text_field, "")).lower().split())
        token_sets.append(tokens)

    selected_indices: List[int] = []
    covered_tokens: set = set()

    # Seed with a random first pick
    first = rng.randint(0, len(examples) - 1)
    selected_indices.append(first)
    covered_tokens.update(token_sets[first])

    while len(selected_indices) < n:
        best_idx = -1
        best_new = -1
        for i in range(len(examples)):
            if i in set(selected_indices):
                continue
            new_tokens = len(token_sets[i] - covered_tokens)
            if new_tokens > best_new:
                best_new = new_tokens
                best_idx = i
        if best_idx < 0:
            break
        selected_indices.append(best_idx)
        covered_tokens.update(token_sets[best_idx])

    return [examples[i] for i in selected_indices]


def select_balanced(
    examples: List[Dict[str, Any]],
    n: int,
    group_field: str = "task_name",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Select N examples with equal representation across groups (task types).

    If a group has fewer examples than its share, the surplus goes to
    the largest groups.
    """
    rng = random.Random(seed)

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for ex in examples:
        key = str(ex.get(group_field, "unknown"))
        groups.setdefault(key, []).append(ex)

    # Shuffle within groups for fairness
    for group_list in groups.values():
        rng.shuffle(group_list)

    num_groups = len(groups)
    if num_groups == 0:
        return []

    per_group = n // num_groups
    remainder = n % num_groups

    selected: List[Dict[str, Any]] = []
    leftover: List[Dict[str, Any]] = []

    for key in sorted(groups.keys()):
        group_list = groups[key]
        take = per_group + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        selected.extend(group_list[:take])
        leftover.extend(group_list[take:])

    # Fill remaining slots from leftover
    if len(selected) < n and leftover:
        rng.shuffle(leftover)
        selected.extend(leftover[: n - len(selected)])

    return selected[:n]


def select_curriculum(
    examples: List[Dict[str, Any]],
    n: int,
    difficulty_field: str = "difficulty",
    ratio: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Select N examples following a curriculum learning distribution.

    Default ratio: 30% easy, 50% medium, 20% hard.
    """
    rng = random.Random(seed)
    ratio = ratio or {"easy": 0.3, "medium": 0.5, "hard": 0.2}

    buckets: Dict[str, List[Dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}
    for ex in examples:
        level = str(ex.get(difficulty_field, "medium"))
        if level not in buckets:
            level = "medium"
        buckets[level].append(ex)

    for bucket_list in buckets.values():
        rng.shuffle(bucket_list)

    selected: List[Dict[str, Any]] = []
    for level, frac in ratio.items():
        take = int(n * frac)
        available = buckets.get(level, [])
        selected.extend(available[:take])

    # Fill remaining slots
    remaining = n - len(selected)
    if remaining > 0:
        all_remaining = []
        for level, bucket_list in buckets.items():
            used = int(n * ratio.get(level, 0))
            all_remaining.extend(bucket_list[used:])
        rng.shuffle(all_remaining)
        selected.extend(all_remaining[:remaining])

    return selected[:n]


def select_examples(
    examples: List[Dict[str, Any]],
    n: int,
    strategy: str = "quality_weighted",
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Unified entry point for example selection.

    Strategies: quality_weighted, diverse, balanced, curriculum.
    """
    strategies = {
        "quality_weighted": select_quality_weighted,
        "diverse": select_diverse,
        "balanced": select_balanced,
        "curriculum": select_curriculum,
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown strategy {strategy!r}. Choose from: {list(strategies.keys())}")

    func = strategies[strategy]
    result = func(examples, n, **kwargs)
    logger.info("Selected %d/%d examples using strategy=%s", len(result), len(examples), strategy)
    return result
