"""Tests for data selection strategies."""

from __future__ import annotations

from training_data_robo.selector import (
    select_balanced,
    select_curriculum,
    select_diverse,
    select_examples,
    select_quality_weighted,
)


def _make_examples(n: int = 20) -> list:
    tasks = ["qa_v1", "summary_v1", "cot_v1", "instruction_v1"]
    examples = []
    for i in range(n):
        examples.append(
            {
                "id": str(i),
                "task_name": tasks[i % len(tasks)],
                "output_text": f"Answer {i} " + ("word " * (i + 5)),
                "judge_avg_score": (i % 5) + 1,  # scores 1-5
                "quality_score": 0.5 + (i % 5) * 0.1,
                "difficulty": ["easy", "medium", "hard"][i % 3],
            }
        )
    return examples


class TestQualityWeighted:
    def test_selects_top_n(self):
        examples = _make_examples(20)
        selected = select_quality_weighted(examples, 5)
        assert len(selected) == 5
        scores = [s["judge_avg_score"] for s in selected]
        assert scores == sorted(scores, reverse=True)

    def test_fallback_to_quality_score(self):
        examples = [{"id": "1", "quality_score": 0.9}, {"id": "2", "quality_score": 0.3}]
        selected = select_quality_weighted(examples, 1)
        assert selected[0]["id"] == "1"

    def test_n_larger_than_available(self):
        examples = _make_examples(3)
        selected = select_quality_weighted(examples, 10)
        assert len(selected) == 3


class TestDiverse:
    def test_selects_n(self):
        examples = _make_examples(20)
        selected = select_diverse(examples, 5)
        assert len(selected) == 5

    def test_diversity_coverage(self):
        # Different examples should cover more tokens than picking first N
        examples = _make_examples(20)
        selected = select_diverse(examples, 10)
        diverse_tokens = set()
        for s in selected:
            diverse_tokens.update(s["output_text"].lower().split())
        # First 10 (non-diverse) would cover fewer unique tokens
        first10_tokens = set()
        for s in examples[:10]:
            first10_tokens.update(s["output_text"].lower().split())
        # Diverse selection should cover at least as many tokens
        assert len(diverse_tokens) >= len(first10_tokens)

    def test_n_larger_than_available(self):
        examples = _make_examples(3)
        selected = select_diverse(examples, 10)
        assert len(selected) == 3


class TestBalanced:
    def test_equal_representation(self):
        examples = _make_examples(20)  # 5 each of 4 task types
        selected = select_balanced(examples, 8)
        assert len(selected) == 8
        tasks = [s["task_name"] for s in selected]
        # Each task should have 2 examples (8/4 = 2)
        from collections import Counter

        counts = Counter(tasks)
        assert all(c == 2 for c in counts.values())

    def test_uneven_groups(self):
        examples = [
            {"id": "1", "task_name": "qa", "output_text": "a"},
            {"id": "2", "task_name": "qa", "output_text": "b"},
            {"id": "3", "task_name": "summary", "output_text": "c"},
        ]
        selected = select_balanced(examples, 2, group_field="task_name")
        assert len(selected) == 2


class TestCurriculum:
    def test_distribution(self):
        examples = _make_examples(30)
        selected = select_curriculum(examples, 10)
        assert len(selected) == 10
        diffs = [s["difficulty"] for s in selected]
        # Should have a mix of difficulties
        assert len(set(diffs)) >= 2

    def test_custom_ratio(self):
        examples = _make_examples(30)
        selected = select_curriculum(examples, 10, ratio={"easy": 0.5, "medium": 0.3, "hard": 0.2})
        assert len(selected) == 10


class TestSelectExamples:
    def test_quality_weighted(self):
        examples = _make_examples(10)
        selected = select_examples(examples, 5, strategy="quality_weighted")
        assert len(selected) == 5

    def test_diverse(self):
        examples = _make_examples(10)
        selected = select_examples(examples, 5, strategy="diverse")
        assert len(selected) == 5

    def test_balanced(self):
        examples = _make_examples(10)
        selected = select_examples(examples, 5, strategy="balanced")
        assert len(selected) == 5

    def test_curriculum(self):
        examples = _make_examples(10)
        selected = select_examples(examples, 5, strategy="curriculum")
        assert len(selected) == 5

    def test_unknown_strategy_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown strategy"):
            select_examples([], 5, strategy="nonexistent")
