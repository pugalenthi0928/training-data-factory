"""Property-based tests for quality filtering and scoring."""
from __future__ import annotations

import string

from hypothesis import given, settings
from hypothesis import strategies as st

from training_data_robo.models import TaskType, TrainingExample
from training_data_robo.quality import QualityRules, deduplicate_examples, filter_examples


def _make_example(
    input_text: str = "What is X?",
    output_text: str = "X is a thing that does stuff.",
    task_name: str = "qa_v1",
) -> TrainingExample:
    return TrainingExample(
        id="test-001",
        task_name=task_name,
        task_type=TaskType.QA,
        input_text=input_text,
        output_text=output_text,
        document_id="doc-1",
        chunk_id="chunk-1",
        model_name="test",
    )


class TestFilterExamplesProperties:
    @given(
        output_len=st.integers(min_value=0, max_value=200),
        min_chars=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50)
    def test_length_filter_consistent(self, output_len: int, min_chars: int) -> None:
        """If output >= min_chars, example should pass length filter."""
        text = "a" * output_len
        ex = _make_example(output_text=text)
        rules = QualityRules(min_output_chars=min_chars, drop_refusals=False)
        result = filter_examples([ex], rules)
        if output_len >= min_chars:
            assert len(result) == 1
        else:
            assert len(result) == 0

    @given(text=st.text(alphabet=string.ascii_letters + " ", min_size=50, max_size=200))
    @settings(max_examples=30)
    def test_non_refusal_text_passes(self, text: str) -> None:
        """Text without refusal markers should pass refusal filter."""
        ex = _make_example(output_text=text)
        rules = QualityRules(min_output_chars=1, drop_refusals=True)
        result = filter_examples([ex], rules)
        assert len(result) == 1

    def test_refusal_markers_caught(self) -> None:
        refusals = [
            "As an AI language model, I cannot do that.",
            "I cannot provide this information.",
            "I'm unable to help with that request.",
        ]
        rules = QualityRules(min_output_chars=1, drop_refusals=True)
        for refusal in refusals:
            result = filter_examples([_make_example(output_text=refusal)], rules)
            assert len(result) == 0, f"Should have filtered: {refusal}"

    def test_empty_list_returns_empty(self) -> None:
        rules = QualityRules()
        assert filter_examples([], rules) == []


class TestDeduplicateProperties:
    @given(n=st.integers(min_value=1, max_value=20))
    @settings(max_examples=20)
    def test_identical_examples_deduplicated(self, n: int) -> None:
        """N identical examples should deduplicate to 1."""
        examples = [_make_example() for _ in range(n)]
        result = deduplicate_examples(examples)
        assert len(result) == 1

    @given(n=st.integers(min_value=1, max_value=20))
    @settings(max_examples=20)
    def test_unique_examples_preserved(self, n: int) -> None:
        """N unique examples should all be preserved."""
        examples = [
            _make_example(input_text=f"Question {i}?", output_text=f"Answer {i}.")
            for i in range(n)
        ]
        result = deduplicate_examples(examples)
        assert len(result) == n

    def test_dedup_preserves_first_occurrence(self) -> None:
        ex1 = _make_example(output_text="First version")
        ex2 = _make_example(output_text="Second version")
        # Same (task_name, input_text) key
        result = deduplicate_examples([ex1, ex2])
        assert len(result) == 1
        assert result[0].output_text == "First version"

    def test_different_tasks_not_deduped(self) -> None:
        ex1 = _make_example(task_name="qa_v1")
        ex2 = _make_example(task_name="summary_v1")
        result = deduplicate_examples([ex1, ex2])
        assert len(result) == 2


class TestQualityRulesDefaults:
    def test_default_rules(self) -> None:
        rules = QualityRules()
        assert rules.min_output_chars == 40
        assert rules.drop_refusals is True
        assert rules.deduplicate is True

    def test_custom_rules(self) -> None:
        rules = QualityRules(min_output_chars=100, drop_refusals=False, deduplicate=False)
        assert rules.min_output_chars == 100
        assert rules.drop_refusals is False
