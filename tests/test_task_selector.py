"""Tests for adaptive task selection."""
from __future__ import annotations

from training_data_robo.models import Document, TaskTemplate, TaskType, TextChunk
from training_data_robo.task_selector import select_tasks_for_chunk


def _make_chunk(text: str, metadata: dict | None = None) -> TextChunk:
    doc = Document.from_text(content="full doc", title="test")
    return TextChunk.from_document(
        document=doc,
        text=text,
        index=0,
        metadata=metadata or {},
    )


def _all_templates() -> list[TaskTemplate]:
    """One template per task type for testing."""
    return [
        TaskTemplate(name="qa", task_type=TaskType.QA, system_prompt="", user_prompt_template="{text}"),
        TaskTemplate(name="summary", task_type=TaskType.SUMMARISATION, system_prompt="", user_prompt_template="{text}"),
        TaskTemplate(name="kp", task_type=TaskType.KEY_POINTS, system_prompt="", user_prompt_template="{text}"),
        TaskTemplate(name="title", task_type=TaskType.TITLE, system_prompt="", user_prompt_template="{text}"),
        TaskTemplate(name="classify", task_type=TaskType.CLASSIFICATION, system_prompt="", user_prompt_template="{text}"),
        TaskTemplate(name="instr", task_type=TaskType.INSTRUCTION_FOLLOWING, system_prompt="", user_prompt_template="{text}"),
        TaskTemplate(name="cot", task_type=TaskType.CHAIN_OF_THOUGHT, system_prompt="", user_prompt_template="{text}"),
    ]


class TestTaskSelector:
    def test_prose_chunk_gets_most_tasks(self):
        chunk = _make_chunk("A " * 200, metadata={"chunk_type": "prose"})
        selected = select_tasks_for_chunk(chunk, _all_templates())
        types = {t.task_type for t in selected}
        assert TaskType.QA in types
        assert TaskType.SUMMARISATION in types
        assert TaskType.INSTRUCTION_FOLLOWING in types
        assert TaskType.CHAIN_OF_THOUGHT in types

    def test_list_chunk_skips_summary_and_cot(self):
        chunk = _make_chunk("- item\n" * 50, metadata={"chunk_type": "list"})
        selected = select_tasks_for_chunk(chunk, _all_templates())
        types = {t.task_type for t in selected}
        assert TaskType.SUMMARISATION not in types
        assert TaskType.CHAIN_OF_THOUGHT not in types
        assert TaskType.INSTRUCTION_FOLLOWING not in types
        assert TaskType.KEY_POINTS in types

    def test_table_chunk_limited(self):
        chunk = _make_chunk("| a | b |\n" * 30, metadata={"chunk_type": "table"})
        selected = select_tasks_for_chunk(chunk, _all_templates())
        types = {t.task_type for t in selected}
        assert TaskType.QA in types
        assert TaskType.CLASSIFICATION in types
        assert TaskType.SUMMARISATION not in types

    def test_short_chunk_filters_by_length(self):
        chunk = _make_chunk("Short.", metadata={"chunk_type": "prose"})
        selected = select_tasks_for_chunk(chunk, _all_templates())
        # Only title has a low enough min_chars (30)
        types = {t.task_type for t in selected}
        assert types == set()  # "Short." is 6 chars, below even title's 30

    def test_medium_chunk_gets_qa_and_title(self):
        chunk = _make_chunk("X " * 60, metadata={"chunk_type": "prose"})
        selected = select_tasks_for_chunk(chunk, _all_templates())
        types = {t.task_type for t in selected}
        assert TaskType.QA in types
        assert TaskType.TITLE in types
        # CLASSIFICATION is not in the prose allowed set
        assert TaskType.CLASSIFICATION not in types
        # 120 chars < 150 required for instruction_following
        assert TaskType.INSTRUCTION_FOLLOWING not in types

    def test_no_metadata_allows_all_by_length(self):
        """Chunks without structure metadata fall back to length-only filtering."""
        chunk = _make_chunk("Y " * 200, metadata={})
        selected = select_tasks_for_chunk(chunk, _all_templates())
        types = {t.task_type for t in selected}
        assert TaskType.QA in types
        assert TaskType.SUMMARISATION in types
        assert TaskType.CHAIN_OF_THOUGHT in types

    def test_empty_templates_returns_empty(self):
        chunk = _make_chunk("A " * 200, metadata={"chunk_type": "prose"})
        assert select_tasks_for_chunk(chunk, []) == []
