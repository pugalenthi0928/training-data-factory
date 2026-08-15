"""Controlled calibration utilities for curation decision thresholds."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .similarity import EmbeddingEncoder, lsh_candidate_pairs, minhash_signature, pair_evidence, word_shingles


@dataclass(frozen=True)
class CalibrationPair:
    pair_id: str
    left: str
    right: str
    duplicate: bool
    category: str


def load_calibration_pairs(path: Path) -> list[CalibrationPair]:
    pairs: list[CalibrationPair] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid calibration JSONL at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Calibration line {line_number} must be an object")
            pairs.append(
                CalibrationPair(
                    pair_id=str(row.get("pair_id", line_number)),
                    left=str(row["left"]),
                    right=str(row["right"]),
                    duplicate=bool(row["duplicate"]),
                    category=str(row.get("category", "unspecified")),
                )
            )
    if not pairs:
        raise ValueError("Calibration fixture is empty")
    if not any(pair.duplicate for pair in pairs) or all(pair.duplicate for pair in pairs):
        raise ValueError("Calibration fixture requires positive and negative pairs")
    return pairs


def binary_metrics(labels: Sequence[bool], predictions: Sequence[bool]) -> dict[str, Any]:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Metrics require equally sized, non-empty labels and predictions")
    tp = sum(label and prediction for label, prediction in zip(labels, predictions, strict=True))
    fp = sum(not label and prediction for label, prediction in zip(labels, predictions, strict=True))
    fn = sum(label and not prediction for label, prediction in zip(labels, predictions, strict=True))
    tn = sum(not label and not prediction for label, prediction in zip(labels, predictions, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def _bootstrap_f1_interval(
    labels: Sequence[bool],
    predictions: Sequence[bool],
    *,
    seeds: Sequence[int],
    resamples_per_seed: int,
) -> dict[str, Any]:
    values: list[float] = []
    for seed in seeds:
        rng = random.Random(seed)
        for _ in range(resamples_per_seed):
            indices = [rng.randrange(len(labels)) for _ in labels]
            sampled_labels = [labels[index] for index in indices]
            sampled_predictions = [predictions[index] for index in indices]
            values.append(float(binary_metrics(sampled_labels, sampled_predictions)["f1"]))
    values.sort()
    lower = values[int(0.025 * (len(values) - 1))]
    upper = values[int(0.975 * (len(values) - 1))]
    return {
        "method": "paired_nonparametric_bootstrap",
        "seeds": list(seeds),
        "resamples_per_seed": resamples_per_seed,
        "confidence": 0.95,
        "f1_lower": lower,
        "f1_upper": upper,
    }


def evaluate_calibration(
    pairs: Sequence[CalibrationPair],
    *,
    fuzzy_thresholds: Sequence[float] = (0.4, 0.5, 0.6, 0.7, 0.8),
    semantic_thresholds: Sequence[float] = (0.8, 0.85, 0.9, 0.95),
    encoder: EmbeddingEncoder | None = None,
    seeds: Sequence[int] = (17, 42, 97),
    resamples_per_seed: int = 200,
) -> dict[str, Any]:
    labels = [pair.duplicate for pair in pairs]
    flat_texts = [text for pair in pairs for text in (pair.left, pair.right)]
    embeddings = encoder.encode(flat_texts) if encoder is not None else None
    evidence = []
    lsh_candidates = []
    for index, pair in enumerate(pairs):
        left_signature = minhash_signature(word_shingles(pair.left, 3))
        right_signature = minhash_signature(word_shingles(pair.right, 3))
        lsh_candidates.append(bool(lsh_candidate_pairs([left_signature, right_signature], bands=32)))
        left_embedding = embeddings[index * 2] if embeddings is not None else None
        right_embedding = embeddings[index * 2 + 1] if embeddings is not None else None
        evidence.append(
            pair_evidence(
                pair.left,
                pair.right,
                shingle_size=3,
                left_embedding=left_embedding,
                right_embedding=right_embedding,
            )
        )

    exact_predictions = [item.exact for item in evidence]
    exact_metrics = binary_metrics(labels, exact_predictions)
    exact_metrics["f1_interval"] = _bootstrap_f1_interval(
        labels, exact_predictions, seeds=seeds, resamples_per_seed=resamples_per_seed
    )

    fuzzy_results = []
    for threshold in fuzzy_thresholds:
        predictions = [
            candidate and item.fuzzy_jaccard >= threshold
            for candidate, item in zip(lsh_candidates, evidence, strict=True)
        ]
        metrics = binary_metrics(labels, predictions)
        metrics.update(
            {
                "threshold": threshold,
                "f1_interval": _bootstrap_f1_interval(
                    labels, predictions, seeds=seeds, resamples_per_seed=resamples_per_seed
                ),
            }
        )
        fuzzy_results.append(metrics)

    semantic_results = []
    if encoder is not None:
        for threshold in semantic_thresholds:
            predictions = [item.semantic_cosine is not None and item.semantic_cosine >= threshold for item in evidence]
            metrics = binary_metrics(labels, predictions)
            metrics.update(
                {
                    "threshold": threshold,
                    "f1_interval": _bootstrap_f1_interval(
                        labels, predictions, seeds=seeds, resamples_per_seed=resamples_per_seed
                    ),
                }
            )
            semantic_results.append(metrics)

    best_fuzzy = max(fuzzy_results, key=lambda item: (float(item["f1"]), float(item["precision"])))
    best_semantic = (
        max(semantic_results, key=lambda item: (float(item["f1"]), float(item["precision"])))
        if semantic_results
        else None
    )
    return {
        "schema_version": "forge.curation-calibration/v1",
        "fixture_pairs": len(pairs),
        "positive_pairs": sum(labels),
        "negative_pairs": len(labels) - sum(labels),
        "categories": sorted({pair.category for pair in pairs}),
        "exact": exact_metrics,
        "fuzzy_minhash_lsh": {"results": fuzzy_results, "selected": best_fuzzy},
        "semantic_embedding": {
            "status": "measured" if encoder is not None else "not_run",
            "model": encoder.name if encoder is not None else None,
            "revision": encoder.revision if encoder is not None else None,
            "results": semantic_results,
            "selected": best_semantic,
        },
        "limitations": [
            "Controlled fixtures measure detector behaviour, not production-corpus prevalence.",
            "Thresholds must be recalibrated when the corpus, language, or embedding model changes.",
        ],
    }
