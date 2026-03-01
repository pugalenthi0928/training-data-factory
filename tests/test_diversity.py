"""Tests for diversity scoring module."""
from __future__ import annotations

from training_data_robo.diversity import compute_diversity_metrics


def _make_examples(n: int, varied: bool = True) -> list:
    examples = []
    tasks = ["qa_v1", "summary_v1", "cot_v1", "instruction_v1"]
    for i in range(n):
        if varied:
            text = f"Answer {i}: " + " ".join(f"word_{i}_{j}" for j in range(20 + i * 5))
            task = tasks[i % len(tasks)]
        else:
            text = "The same exact answer repeated verbatim for every single example"
            task = "qa_v1"
        examples.append({
            "id": str(i),
            "task_name": task,
            "output_text": text,
        })
    return examples


class TestDiversityMetrics:
    def test_empty(self):
        m = compute_diversity_metrics([])
        assert m["num_examples"] == 0
        assert m["vocab_diversity"] == 0.0

    def test_basic_metrics(self):
        examples = _make_examples(10, varied=True)
        m = compute_diversity_metrics(examples)
        assert m["num_examples"] == 10
        assert m["task_coverage"] == 4
        assert m["vocab_diversity"] > 0
        assert m["avg_output_length"] > 0

    def test_redundant_dataset(self):
        examples = _make_examples(10, varied=False)
        m = compute_diversity_metrics(examples)
        # All same text → high redundancy
        assert m["redundancy_ratio"] > 0.5
        assert m["task_coverage"] == 1

    def test_diverse_dataset(self):
        examples = _make_examples(10, varied=True)
        m = compute_diversity_metrics(examples)
        # Different texts → low redundancy
        assert m["redundancy_ratio"] < 0.3

    def test_task_distribution(self):
        examples = _make_examples(8, varied=True)
        m = compute_diversity_metrics(examples)
        assert "task_distribution" in m
        assert sum(m["task_distribution"].values()) == 8

    def test_length_cv(self):
        examples = _make_examples(10, varied=True)
        m = compute_diversity_metrics(examples)
        # Varied examples have different lengths → non-zero CV
        assert m["length_cv"] > 0
