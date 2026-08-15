"""Adaptive task selection based on chunk metadata.

Given a TextChunk with structure-aware metadata (chunk_type, section_title, etc.),
select the most appropriate subset of task templates to apply. This avoids wasting
LLM calls on tasks that do not make sense for a given chunk type, such as asking for
a summary of a 3-row table, or key-points extraction from a title-only chunk.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from .models import ChunkType, TaskTemplate, TaskType, TextChunk

# Minimum text length (chars) to bother running a task
_MIN_CHARS_FOR_TASK: Dict[TaskType, int] = {
    TaskType.SUMMARISATION: 200,
    TaskType.KEY_POINTS: 150,
    TaskType.QA: 80,
    TaskType.TITLE: 30,
    TaskType.CLASSIFICATION: 50,
    TaskType.INSTRUCTION_FOLLOWING: 150,
    TaskType.CHAIN_OF_THOUGHT: 200,
}

# Which task types are suitable for each chunk type
_CHUNK_TYPE_TASKS: Dict[str, List[TaskType]] = {
    ChunkType.PROSE.value: [
        TaskType.QA,
        TaskType.SUMMARISATION,
        TaskType.KEY_POINTS,
        TaskType.TITLE,
        TaskType.INSTRUCTION_FOLLOWING,
        TaskType.CHAIN_OF_THOUGHT,
    ],
    ChunkType.LIST.value: [
        TaskType.KEY_POINTS,
        TaskType.CLASSIFICATION,
        TaskType.QA,
        TaskType.TITLE,
    ],
    ChunkType.TABLE.value: [
        TaskType.QA,
        TaskType.CLASSIFICATION,
        TaskType.TITLE,
    ],
    ChunkType.MIXED.value: [
        TaskType.QA,
        TaskType.SUMMARISATION,
        TaskType.KEY_POINTS,
        TaskType.TITLE,
        TaskType.INSTRUCTION_FOLLOWING,
    ],
}


def select_tasks_for_chunk(
    chunk: TextChunk,
    available_templates: Sequence[TaskTemplate],
) -> List[TaskTemplate]:
    """Return the subset of templates that are appropriate for this chunk.

    Selection rules:
    1. Skip tasks if chunk text is below the minimum length for that task type.
    2. Skip tasks that don't match the chunk's content type (prose/list/table/mixed).
    3. If chunk has no structure metadata, allow all tasks (fallback to length filter only).
    """
    text_len = len(chunk.text.strip())
    chunk_type = chunk.metadata.get("chunk_type")

    # Determine allowed task types for this chunk type
    if chunk_type and chunk_type in _CHUNK_TYPE_TASKS:
        allowed_types = set(_CHUNK_TYPE_TASKS[chunk_type])
    else:
        # No structure metadata, so allow every task.
        allowed_types = set(TaskType)

    selected: List[TaskTemplate] = []
    for tmpl in available_templates:
        # Check chunk type compatibility
        if tmpl.task_type not in allowed_types:
            continue

        # Check minimum length
        min_chars = _MIN_CHARS_FOR_TASK.get(tmpl.task_type, 50)
        if text_len < min_chars:
            continue

        selected.append(tmpl)

    return selected
