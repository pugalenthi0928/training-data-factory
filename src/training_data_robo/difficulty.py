"""Heuristic difficulty calibration for training examples.

Tags each example as easy/medium/hard based on:
  - Output length (longer answers = harder)
  - Vocabulary complexity (rare words, technical terms)
  - Reasoning indicators (step-by-step, because, therefore, etc.)
  - Question complexity (multi-part questions, comparison questions)
"""
from __future__ import annotations

import re
import string
from typing import Any, Dict, List

from .models import DifficultyLevel

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# Words that signal multi-step reasoning
_REASONING_MARKERS = {
    "therefore", "because", "however", "furthermore", "consequently",
    "moreover", "nevertheless", "although", "whereas", "implies",
    "hence", "thus", "assuming", "given that", "it follows",
    "step 1", "step 2", "step 3", "first", "second", "third",
    "finally", "in conclusion", "to summarize",
}

# Words that signal comparison / complex questions
_COMPLEXITY_MARKERS = {
    "compare", "contrast", "difference", "relationship", "analyze",
    "evaluate", "explain why", "how does", "what would happen",
    "implications", "trade-off", "tradeoff", "advantages", "disadvantages",
}


def _count_reasoning_signals(text: str) -> int:
    lower = text.lower()
    count = 0
    for marker in _REASONING_MARKERS:
        if marker in lower:
            count += 1
    return count


def _count_complexity_signals(text: str) -> int:
    lower = text.lower()
    count = 0
    for marker in _COMPLEXITY_MARKERS:
        if marker in lower:
            count += 1
    return count


def _vocab_complexity(text: str) -> float:
    """Fraction of words >= 8 characters (proxy for technical vocabulary)."""
    words = text.lower().translate(_PUNCT_TABLE).split()
    if not words:
        return 0.0
    long_words = sum(1 for w in words if len(w) >= 8)
    return long_words / len(words)


def _sentence_count(text: str) -> int:
    """Approximate sentence count."""
    return len(re.split(r"[.!?]+", text.strip())) - 1 or 1


def calibrate_difficulty(example: Dict[str, Any]) -> Dict[str, Any]:
    """Tag a single example with a difficulty level and feature scores.

    Returns the example dict updated with 'difficulty' and 'difficulty_features'.
    """
    output = str(example.get("output_text", ""))
    input_text = str(example.get("input_text", ""))
    combined = input_text + " " + output

    # Feature extraction
    out_len = len(output)
    out_sentences = _sentence_count(output)
    reasoning = _count_reasoning_signals(combined)
    complexity = _count_complexity_signals(combined)
    vocab = _vocab_complexity(output)

    # Scoring: accumulate difficulty points
    score = 0.0

    # Output length
    if out_len > 500:
        score += 2.0
    elif out_len > 200:
        score += 1.0

    # Sentence count
    if out_sentences > 8:
        score += 1.5
    elif out_sentences > 4:
        score += 0.5

    # Reasoning signals
    if reasoning >= 3:
        score += 2.0
    elif reasoning >= 1:
        score += 1.0

    # Complexity signals
    if complexity >= 2:
        score += 1.5
    elif complexity >= 1:
        score += 0.5

    # Vocabulary complexity
    if vocab > 0.15:
        score += 1.0
    elif vocab > 0.08:
        score += 0.5

    # Map score to difficulty level
    if score >= 4.0:
        level = DifficultyLevel.HARD
    elif score >= 2.0:
        level = DifficultyLevel.MEDIUM
    else:
        level = DifficultyLevel.EASY

    example["difficulty"] = level.value
    example["difficulty_features"] = {
        "output_length": out_len,
        "sentence_count": out_sentences,
        "reasoning_signals": reasoning,
        "complexity_signals": complexity,
        "vocab_complexity": round(vocab, 4),
        "raw_score": round(score, 2),
    }
    return example


def calibrate_batch(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tag all examples with difficulty and return summary stats."""
    for ex in examples:
        calibrate_difficulty(ex)

    distribution = {"easy": 0, "medium": 0, "hard": 0}
    for ex in examples:
        level = ex.get("difficulty", "easy")
        distribution[level] = distribution.get(level, 0) + 1

    return {
        "total": len(examples),
        "distribution": distribution,
    }
