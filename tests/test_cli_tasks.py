"""Tests for CLI task template building including new task types."""
from __future__ import annotations

import pytest

from training_data_robo.cli import build_task_templates_from_names
from training_data_robo.models import TaskType


class TestBuildTaskTemplates:
    def test_qa(self):
        templates = build_task_templates_from_names(["qa"])
        assert len(templates) == 1
        assert templates[0].task_type == TaskType.QA

    def test_summary(self):
        templates = build_task_templates_from_names(["summary"])
        assert len(templates) == 1
        assert templates[0].task_type == TaskType.SUMMARISATION

    def test_instruction(self):
        templates = build_task_templates_from_names(["instruction"])
        assert len(templates) == 1
        assert templates[0].task_type == TaskType.INSTRUCTION_FOLLOWING
        assert "instruction" in templates[0].name.lower()

    def test_instruction_following_alias(self):
        templates = build_task_templates_from_names(["instruction_following"])
        assert len(templates) == 1
        assert templates[0].task_type == TaskType.INSTRUCTION_FOLLOWING

    def test_cot(self):
        templates = build_task_templates_from_names(["cot"])
        assert len(templates) == 1
        assert templates[0].task_type == TaskType.CHAIN_OF_THOUGHT
        assert "REASONING" in templates[0].user_prompt_template

    def test_chain_of_thought_alias(self):
        templates = build_task_templates_from_names(["chain_of_thought"])
        assert len(templates) == 1
        assert templates[0].task_type == TaskType.CHAIN_OF_THOUGHT

    def test_multiple_tasks(self):
        templates = build_task_templates_from_names(["qa", "summary", "instruction", "cot"])
        assert len(templates) == 4
        types = {t.task_type for t in templates}
        assert types == {
            TaskType.QA,
            TaskType.SUMMARISATION,
            TaskType.INSTRUCTION_FOLLOWING,
            TaskType.CHAIN_OF_THOUGHT,
        }

    def test_all_six_tasks(self):
        templates = build_task_templates_from_names(
            ["qa", "summary", "key_points", "title", "instruction", "cot"]
        )
        assert len(templates) == 6

    def test_invalid_task_raises(self):
        with pytest.raises(SystemExit):
            build_task_templates_from_names(["nonexistent_task"])

    def test_empty_raises(self):
        with pytest.raises(SystemExit):
            build_task_templates_from_names([])
