"""Versioned pairwise judge runner for blinded Forge evaluation packets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import canonical_sha256
from .evaluation import ANNOTATION_REASON_CODES, EvaluationValidationError

PAIRWISE_JUDGE_PROMPT_VERSION = "forge.pairwise-judge/v1"
PAIRWISE_JUDGE_SYSTEM = (
    "You are evaluating two candidate training examples without knowing which system produced them. "
    "Use only the supplied source, task, prompt, and reference. Prefer material correctness, source support, "
    "instruction fit, and training usefulness over style or length. Return only JSON with preference equal to "
    "A, B, tie, or both_bad; confidence from 0 to 1; a short explanation; and zero or more reason codes from: "
    + ", ".join(ANNOTATION_REASON_CODES)
    + "."
)
PAIRWISE_JUDGE_PROMPT_SHA256 = canonical_sha256(
    {"version": PAIRWISE_JUDGE_PROMPT_VERSION, "system": PAIRWISE_JUDGE_SYSTEM}
)


def build_pairwise_prompt(presentation: Mapping[str, Any]) -> str:
    """Render the stable user prompt for one blinded presentation."""
    required = (
        "presentation_id",
        "task",
        "source_excerpt",
        "prompt",
        "reference_answer",
        "candidate_a",
        "candidate_b",
    )
    missing = [field for field in required if not str(presentation.get(field, "")).strip()]
    if missing:
        raise EvaluationValidationError(f"judge presentation is missing: {', '.join(missing)}")
    return (
        f"## Source\n{presentation['source_excerpt']}\n\n"
        f"## Task\n{presentation['task']}\n\n"
        f"## Prompt\n{presentation['prompt']}\n\n"
        f"## Reference\n{presentation['reference_answer']}\n\n"
        f"## Candidate A\n{presentation['candidate_a']}\n\n"
        f"## Candidate B\n{presentation['candidate_b']}\n\n"
        "Return JSON only."
    )


def parse_pairwise_judgment(raw: str) -> dict[str, Any]:
    """Parse strict judge JSON without inventing a fallback decision."""
    value = raw.strip()
    if value.startswith("```"):
        start = value.find("{")
        end = value.rfind("}") + 1
        value = value[start:end] if start >= 0 and end > start else value
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise EvaluationValidationError("pairwise judge returned invalid JSON") from exc
    if not isinstance(data, dict) or data.get("preference") not in {"A", "B", "tie", "both_bad"}:
        raise EvaluationValidationError("pairwise judge returned an invalid preference")
    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        raise EvaluationValidationError("pairwise judge returned invalid confidence")
    reason_codes = data.get("reason_codes", [])
    if not isinstance(reason_codes, list) or not all(isinstance(code, str) for code in reason_codes):
        raise EvaluationValidationError("pairwise judge returned invalid reason codes")
    if any(code not in ANNOTATION_REASON_CODES for code in reason_codes):
        raise EvaluationValidationError("pairwise judge returned an unknown reason code")
    return {
        "preference": str(data["preference"]),
        "confidence": round(float(confidence), 6),
        "explanation": str(data.get("explanation", ""))[:500],
        "reason_codes": reason_codes,
    }


def run_pairwise_judge(
    packet_path: Path,
    output_path: Path,
    *,
    model: str,
    judge_family: str,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Evaluate a frozen judge packet with an OpenAI Responses model."""
    if not model.strip() or not judge_family.strip():
        raise EvaluationValidationError("judge model and family are required")
    from openai import OpenAI

    client = OpenAI(api_key=api_key) if api_key else OpenAI()
    predictions: list[dict[str, Any]] = []
    for line_number, line in enumerate(packet_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            presentation = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationValidationError(f"judge packet has invalid JSON on line {line_number}") from exc
        if not isinstance(presentation, dict):
            raise EvaluationValidationError(f"judge packet line {line_number} must be a JSON object")
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": PAIRWISE_JUDGE_SYSTEM},
                {"role": "user", "content": build_pairwise_prompt(presentation)},
            ],
            max_output_tokens=250,
            temperature=0.0,
        )
        parsed = parse_pairwise_judgment(response.output_text)
        predictions.append(
            {
                "schema_version": "forge.judge-prediction/v1",
                "presentation_id": str(presentation["presentation_id"]),
                **parsed,
                "provider": "openai",
                "judge_model": model,
                "judge_family": judge_family,
                "prompt_version": PAIRWISE_JUDGE_PROMPT_VERSION,
                "prompt_sha256": PAIRWISE_JUDGE_PROMPT_SHA256,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(prediction, ensure_ascii=False) + "\n" for prediction in predictions),
        encoding="utf-8",
    )
    return predictions
