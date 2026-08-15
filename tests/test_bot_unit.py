"""Unit tests for TrainingDataBot internals."""

from __future__ import annotations

from training_data_robo.bot import TrainingDataBot
from training_data_robo.models import TaskType, TrainingExample


def _make_example(
    input_text: str = "What is X?",
    output_text: str = "X is a detailed answer that explains the concept thoroughly.",
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


class TestFilterAndDedup:
    def test_keeps_good_examples(self) -> None:
        bot = TrainingDataBot()
        examples = [
            _make_example(output_text="This is a perfectly good answer that should pass all filters."),
        ]
        result = bot._filter_and_dedup_examples(examples)
        assert len(result) == 1

    def test_drops_short_outputs(self) -> None:
        bot = TrainingDataBot()
        examples = [_make_example(output_text="Too short")]
        result = bot._filter_and_dedup_examples(examples)
        assert len(result) == 0

    def test_drops_refusals(self) -> None:
        bot = TrainingDataBot()
        refusal_examples = [
            _make_example(output_text="As an AI language model, I cannot help with that request and must decline."),
            _make_example(output_text="I cannot provide this information to you at this time for safety reasons."),
            _make_example(output_text="I'm unable to assist with that particular request you've made."),
        ]
        result = bot._filter_and_dedup_examples(refusal_examples)
        assert len(result) == 0

    def test_deduplicates_identical(self) -> None:
        bot = TrainingDataBot()
        ex = _make_example(output_text="This is a sufficiently long answer for the quality filter.")
        examples = [ex, ex, ex]
        result = bot._filter_and_dedup_examples(examples)
        assert len(result) == 1

    def test_keeps_different_tasks(self) -> None:
        bot = TrainingDataBot()
        examples = [
            _make_example(task_name="qa_v1", output_text="A detailed answer about question answering and its methods."),
            _make_example(
                task_name="summary_v1", output_text="A detailed answer about question answering and its methods."
            ),
        ]
        result = bot._filter_and_dedup_examples(examples)
        assert len(result) == 2

    def test_keeps_different_inputs(self) -> None:
        bot = TrainingDataBot()
        examples = [
            _make_example(
                input_text="Question A?", output_text="Answer to question A with enough detail to pass filter."
            ),
            _make_example(
                input_text="Question B?", output_text="Answer to question B with enough detail to pass filter."
            ),
        ]
        result = bot._filter_and_dedup_examples(examples)
        assert len(result) == 2

    def test_mixed_quality(self) -> None:
        bot = TrainingDataBot()
        examples = [
            _make_example(output_text="Good answer with sufficient length for the quality filter check."),
            _make_example(output_text="short"),
            _make_example(output_text="As an AI language model, I cannot answer that question for you today."),
            _make_example(output_text="Another good answer with plenty of detail and explanation inside."),
        ]
        result = bot._filter_and_dedup_examples(examples)
        assert len(result) == 2

    def test_empty_input(self) -> None:
        bot = TrainingDataBot()
        result = bot._filter_and_dedup_examples([])
        assert result == []
