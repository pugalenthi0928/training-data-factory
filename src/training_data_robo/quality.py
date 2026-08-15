from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Set, Tuple

from .models import TrainingExample

REFUSAL_MARKERS = [
    "as an ai language model",
    "i cannot",
    "i'm unable to",
    "cannot help with that request",
]


@dataclass
class QualityRules:
    min_output_chars: int = 40
    drop_refusals: bool = True
    deduplicate: bool = True  # identical (task_name, input_text) treated as dup


def filter_examples(examples: Iterable[TrainingExample], rules: QualityRules) -> List[TrainingExample]:
    cleaned: List[TrainingExample] = []
    for ex in examples:
        out = (ex.output_text or "").strip()
        if rules.min_output_chars and len(out) < rules.min_output_chars:
            continue
        if rules.drop_refusals:
            low = out.lower()
            if any(m in low for m in REFUSAL_MARKERS):
                continue
        cleaned.append(ex)
    return cleaned


def deduplicate_examples(examples: Iterable[TrainingExample]) -> List[TrainingExample]:
    seen: Set[Tuple[str, str]] = set()
    unique: List[TrainingExample] = []
    for ex in examples:
        key = (ex.task_name, ex.input_text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(ex)
    return unique
