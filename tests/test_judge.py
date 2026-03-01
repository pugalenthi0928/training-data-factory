"""Tests for LLM-as-Judge module."""
from __future__ import annotations

import asyncio

from training_data_robo.judge import (
    DummyJudge,
    JudgeResult,
    _parse_verdict,
)
from training_data_robo.models import JudgeRubric


def _make_example(output_text: str = "A detailed answer.", example_id: str = "ex1") -> dict:
    return {
        "id": example_id,
        "task_name": "qa_v1",
        "input_text": "What is machine learning?",
        "output_text": output_text,
    }


class TestParseVerdict:
    def test_valid_json(self):
        raw = '{"score": 4, "explanation": "Good answer."}'
        v = _parse_verdict(raw, "faithfulness")
        assert v.dimension == "faithfulness"
        assert v.score == 4
        assert v.explanation == "Good answer."

    def test_json_in_code_block(self):
        raw = '```json\n{"score": 5, "explanation": "Excellent."}\n```'
        v = _parse_verdict(raw, "coherence")
        assert v.score == 5

    def test_score_clamped_high(self):
        raw = '{"score": 99, "explanation": "Over the top."}'
        v = _parse_verdict(raw, "helpfulness")
        assert v.score == 5

    def test_score_clamped_low(self):
        raw = '{"score": -1, "explanation": "Too low."}'
        v = _parse_verdict(raw, "complexity")
        assert v.score == 1

    def test_malformed_json(self):
        raw = "this is not json at all"
        v = _parse_verdict(raw, "faithfulness")
        assert v.score == 3  # fallback
        assert "parse error" in v.explanation


class TestDummyJudge:
    def test_judge_single(self):
        judge = DummyJudge()
        ex = _make_example("A" * 200)
        result = asyncio.run(judge.judge_example(ex))
        assert isinstance(result, JudgeResult)
        assert len(result.verdicts) == 4  # default rubric has 4 dims
        assert result.avg_score > 0

    def test_judge_batch(self):
        judge = DummyJudge()
        examples = [_make_example(f"Answer {i}" * 30, f"ex{i}") for i in range(5)]
        results = asyncio.run(judge.judge_batch(examples))
        assert len(results) == 5
        for r in results:
            assert r.example_id.startswith("ex")
            assert len(r.verdicts) == 4

    def test_score_varies_by_length(self):
        judge = DummyJudge()
        short = asyncio.run(judge.judge_example(_make_example("Hi")))
        long = asyncio.run(judge.judge_example(_make_example("X" * 300)))
        assert long.avg_score >= short.avg_score

    def test_to_dict(self):
        judge = DummyJudge()
        result = asyncio.run(judge.judge_example(_make_example()))
        d = result.to_dict()
        assert "verdicts" in d
        assert "avg_score" in d
        assert "faithfulness" in d["verdicts"]

    def test_custom_rubric(self):
        rubric = JudgeRubric(dimensions=[
            __import__("training_data_robo.models", fromlist=["QualityDimension"]).QualityDimension(
                name="test_dim", description="Test dimension"
            )
        ])
        judge = DummyJudge(rubric=rubric)
        result = asyncio.run(judge.judge_example(_make_example()))
        assert len(result.verdicts) == 1
        assert result.verdicts[0].dimension == "test_dim"
