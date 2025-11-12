from __future__ import annotations

from typing import List, Sequence, Optional

from .ai_client import BaseLLMClient
from .models import TextChunk, TaskTemplate, TrainingExample


class TaskManager:
    """
    Orchestrates applying TaskTemplate(s) to text chunks using an LLM client.
    """

    def __init__(self, llm_client: BaseLLMClient) -> None:
        self.llm_client = llm_client

    async def run_tasks_on_chunks(
        self,
        chunks: Sequence[TextChunk],
        task_templates: Sequence[TaskTemplate],
        max_examples: Optional[int] = None,
    ) -> List[TrainingExample]:
        examples: List[TrainingExample] = []
        model_name = getattr(self.llm_client, "model", "unknown")

        stop = False

        for chunk in chunks:
            for template in task_templates:
                if max_examples is not None and len(examples) >= max_examples:
                    stop = True
                    break

                user_prompt = template.user_prompt_template.format(text=chunk.text)

                output = await self.llm_client.generate(
                    system_prompt=template.system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=template.max_output_tokens,
                    temperature=template.temperature,
                    top_p=template.top_p,
                )

                examples.append(
                    TrainingExample(
                        id=str(len(examples) + 1),
                        task_name=template.name,
                        task_type=template.task_type,
                        input_text=user_prompt,
                        output_text=output,
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        model_name=model_name,
                        task_version=template.version,
                        temperature=template.temperature,
                    )
                )

            if stop:
                break

        return examples
