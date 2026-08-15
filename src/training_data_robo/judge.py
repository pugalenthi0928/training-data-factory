"""LLM-as-Judge: score training examples on rubric dimensions.

Uses GPT-4.1-mini to evaluate each training example on 4 quality
dimensions (faithfulness, helpfulness, complexity, coherence) with
explicit scoring rubrics. Each verdict includes a 1-5 score,
one-sentence explanation, and cost tracking.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .logging_config import get_logger
from .models import JudgeRubric, QualityDimension

logger = get_logger("training_data_robo.judge")

_JUDGE_SYSTEM = (
    "You are an expert training-data quality judge. "
    "Score the assistant's output on the given dimension using a 1-5 scale. "
    'Respond with ONLY valid JSON: {"score": <int 1-5>, "explanation": "<one sentence>"}'
)

_JUDGE_USER_TMPL = (
    "## Dimension: {dim_name}\n"
    "{dim_description}\n\n"
    "Score scale: {min_score} (worst) to {max_score} (best)\n\n"
    "## Source text (context)\n{context}\n\n"
    "## Task: {task_name}\n\n"
    "## Input\n{input_text}\n\n"
    "## Output to judge\n{output_text}\n\n"
    'Return JSON: {{"score": <int>, "explanation": "<one sentence>"}}'
)


@dataclass
class JudgeVerdict:
    dimension: str
    score: int
    explanation: str
    raw_response: str = ""


@dataclass
class JudgeResult:
    example_id: str
    verdicts: List[JudgeVerdict] = field(default_factory=list)
    avg_score: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "verdicts": {v.dimension: {"score": v.score, "explanation": v.explanation} for v in self.verdicts},
            "avg_score": self.avg_score,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }


def _parse_verdict(raw: str, dimension: str) -> JudgeVerdict:
    """Parse the LLM's JSON verdict, with fallback for malformed responses."""
    raw = raw.strip()
    # Try to extract JSON from the response
    try:
        # Handle case where LLM wraps in markdown code block
        if "```" in raw:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                raw = raw[start:end]
        data = json.loads(raw)
        score = int(data.get("score", 3))
        score = max(1, min(5, score))
        explanation = str(data.get("explanation", ""))
        return JudgeVerdict(dimension=dimension, score=score, explanation=explanation, raw_response=raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning("Failed to parse judge response for %s: %s", dimension, raw[:200])
        return JudgeVerdict(dimension=dimension, score=3, explanation="(parse error)", raw_response=raw)


class LLMJudge:
    """Score training examples using an LLM judge with rubric-based evaluation."""

    def __init__(
        self,
        rubric: Optional[JudgeRubric] = None,
        model: str = "gpt-4.1-mini",
        api_key: Optional[str] = None,
        max_concurrent: int = 5,
    ) -> None:
        self.rubric = rubric or JudgeRubric()
        self.model = model
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Lazy-init OpenAI client
        self._api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            if self._api_key:
                self._client = OpenAI(api_key=self._api_key)
            else:
                self._client = OpenAI()
        return self._client

    async def _score_dimension(
        self,
        dim: QualityDimension,
        example: Dict[str, Any],
        context: str,
    ) -> JudgeVerdict:
        """Score a single example on a single dimension."""
        user_prompt = _JUDGE_USER_TMPL.format(
            dim_name=dim.name,
            dim_description=dim.description,
            min_score=dim.min_score,
            max_score=dim.max_score,
            context=context[:2000],  # truncate to avoid token overflow
            task_name=example.get("task_name", "unknown"),
            input_text=str(example.get("input_text", ""))[:1500],
            output_text=str(example.get("output_text", ""))[:1500],
        )

        async with self._semaphore:
            try:
                client = self._get_client()
                response = client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": _JUDGE_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_output_tokens=150,
                    temperature=0.1,
                )
                raw = response.output_text
            except Exception as e:
                logger.warning("Judge API call failed for %s/%s: %s", example.get("id"), dim.name, e)
                raw = json.dumps({"score": 3, "explanation": f"API error: {e}"})

        return _parse_verdict(raw, dim.name)

    async def judge_example(
        self,
        example: Dict[str, Any],
        context: str = "",
    ) -> JudgeResult:
        """Score one example on all rubric dimensions."""
        tasks = [self._score_dimension(dim, example, context) for dim in self.rubric.dimensions]
        verdicts = await asyncio.gather(*tasks)

        scores = [v.score for v in verdicts]
        avg = sum(scores) / len(scores) if scores else 0.0

        return JudgeResult(
            example_id=str(example.get("id", "")),
            verdicts=list(verdicts),
            avg_score=round(avg, 2),
        )

    async def judge_batch(
        self,
        examples: List[Dict[str, Any]],
        context_field: str = "input_text",
    ) -> List[JudgeResult]:
        """Score a batch of examples."""
        logger.info("Judging %d examples on %d dimensions...", len(examples), len(self.rubric.dimensions))
        results = []
        for i, ex in enumerate(examples):
            context = str(ex.get(context_field, ""))
            result = await self.judge_example(ex, context=context)
            results.append(result)
            if (i + 1) % 10 == 0:
                logger.info("Judged %d/%d examples", i + 1, len(examples))
        logger.info("Judging complete: %d examples scored", len(results))
        return results


class DummyJudge:
    """Deterministic mock judge for testing (no API calls)."""

    def __init__(self, rubric: Optional[JudgeRubric] = None) -> None:
        self.rubric = rubric or JudgeRubric()

    async def judge_example(
        self,
        example: Dict[str, Any],
        context: str = "",
    ) -> JudgeResult:
        verdicts = []
        for dim in self.rubric.dimensions:
            # Deterministic scoring based on output length
            out_len = len(str(example.get("output_text", "")))
            score = min(5, max(1, out_len // 50 + 1))
            verdicts.append(
                JudgeVerdict(
                    dimension=dim.name,
                    score=score,
                    explanation=f"Dummy score based on output length ({out_len} chars)",
                )
            )
        scores = [v.score for v in verdicts]
        return JudgeResult(
            example_id=str(example.get("id", "")),
            verdicts=verdicts,
            avg_score=round(sum(scores) / len(scores), 2) if scores else 0.0,
        )

    async def judge_batch(
        self,
        examples: List[Dict[str, Any]],
        context_field: str = "input_text",
    ) -> List[JudgeResult]:
        return [await self.judge_example(ex) for ex in examples]
