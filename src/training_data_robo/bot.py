from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from .ai_client import BaseLLMClient, DummyLLMClient, OpenAILLMClient
from .chunking import simple_chunk_document as chunk_document
from .errors import TrainingDataBotError
from .logging_config import get_logger
from .models import (
    Dataset,
    Document,
    TaskTemplate,
    TextChunk,
    TrainingExample,
    new_dataset,
)
from .settings import Settings
from .sources import UnifiedLoader
from .tasks import TaskManager


class TrainingDataBot:
    """
    High-level orchestration object for the Training Data Robo factory.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.logger = get_logger("training_data_robo.bot")

        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, TextChunk] = {}
        self.datasets: Dict[str, Dataset] = {}

        self.loader = UnifiedLoader()
        self.llm_client: BaseLLMClient

        if self.settings.openai_api_key:
            self.llm_client = OpenAILLMClient(
                model=self.settings.default_model,
                api_key=self.settings.openai_api_key,
            )
            self.logger.info(
                "OPENAI_API_KEY detected; using OpenAILLMClient with model %s",
                self.settings.default_model,
            )
        else:
            self.llm_client = DummyLLMClient()
            self.logger.info("No OPENAI_API_KEY set; using DummyLLMClient (no external calls).")

        self.task_manager = TaskManager(self.llm_client)
        self.logger.info("TrainingDataBot initialized successfully.")

    async def __aenter__(self) -> "TrainingDataBot":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.logger.info("TrainingDataBot cleanup complete.")

    # ---------- Document loading & chunking ----------

    async def load_documents(
        self,
        sources: Iterable[Union[str, Path]],
    ) -> None:
        sources_list = list(sources)
        self.logger.info("Loading documents from %d sources...", len(sources_list))
        docs = await self.loader.load_documents(sources_list)
        self.documents = {d.id: d for d in docs}
        self.logger.info("Loaded %d documents.", len(self.documents))

    async def chunk_documents(
        self,
        max_chars: int = 800,
        overlap: int = 100,
    ) -> None:
        if not self.documents:
            raise TrainingDataBotError("No documents loaded. Did you call load_documents()?")

        self.logger.info(
            "Chunking %d documents (max_chars=%d, overlap=%d)...",
            len(self.documents),
            max_chars,
            overlap,
        )

        all_chunks: Dict[str, TextChunk] = {}
        for doc in self.documents.values():
            doc_chunks = chunk_document(doc, max_chars=max_chars, overlap=overlap)
            for c in doc_chunks:
                all_chunks[c.id] = c

        self.chunks = all_chunks
        self.logger.info(
            "Created %d chunks from %d documents.",
            len(self.chunks),
            len(self.documents),
        )

    # ---------- Processing & datasets ----------

    async def process_documents(
        self,
        tasks: Optional[List[TaskTemplate]] = None,
        max_examples: Optional[int] = None,
    ) -> str:
        if not tasks:
            raise TrainingDataBotError("No tasks provided to process_documents().")

        if not self.chunks:
            raise TrainingDataBotError("No chunks found. Did you call chunk_documents()?")

        chunks_list = list(self.chunks.values())
        self.logger.info(
            "Processing %d chunks with %d task templates...",
            len(chunks_list),
            len(tasks),
        )

        examples: List[TrainingExample] = await self.task_manager.run_tasks_on_chunks(
            chunks_list,
            tasks,
            max_examples=max_examples,
        )

        # Quality filter + dedup
        examples = self._filter_and_dedup_examples(examples)

        dataset = new_dataset(name="default")
        dataset.examples.extend(examples)
        self.datasets[dataset.id] = dataset

        self.logger.info(
            "Created dataset %s with %d examples.",
            dataset.id,
            len(dataset.examples),
        )
        return dataset.id

    def _filter_and_dedup_examples(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """
        Heuristic quality filter:
          - drop very short outputs (configurable via settings.min_output_chars)
          - drop obvious refusals (configurable via settings.drop_refusals)
          - deduplicate identical (task_name, input_text, output_text) tuples
            (configurable via settings.deduplicate)
        """
        min_output_chars: int = getattr(self.settings, "min_output_chars", 40)
        drop_refusals: bool = getattr(self.settings, "drop_refusals", True)
        deduplicate: bool = getattr(self.settings, "deduplicate", True)

        refusal_markers = [
            "as an ai language model",
            "as an artificial intelligence",
            "i cannot",
            "i'm unable to",
            "i can’t",
            "cannot help with that request",
        ]

        filtered: List[TrainingExample] = []
        seen = set()

        dropped_short = 0
        dropped_refusal = 0
        dropped_dup = 0

        for ex in examples:
            out = (ex.output_text or "").strip()
            if min_output_chars and len(out) < min_output_chars:
                dropped_short += 1
                continue

            lower = out.lower()
            if drop_refusals and any(marker in lower for marker in refusal_markers):
                dropped_refusal += 1
                continue

            if deduplicate:
                key = (ex.task_name, ex.input_text.strip(), out)
                if key in seen:
                    dropped_dup += 1
                    continue
                seen.add(key)

            filtered.append(ex)

        total_dropped = dropped_short + dropped_refusal + dropped_dup
        if total_dropped:
            self.logger.info(
                "Quality filter dropped %d examples (short=%d, refusals=%d, dups=%d)",
                total_dropped,
                dropped_short,
                dropped_refusal,
                dropped_dup,
            )

        return filtered

    async def evaluate_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """
        Basic dataset quality / stats summary.
        """
        dataset = self.datasets.get(dataset_id)
        if dataset is None:
            raise TrainingDataBotError(f"Unknown dataset_id: {dataset_id!r}")

        num_examples = len(dataset.examples)
        if num_examples == 0:
            return {
                "num_examples": 0,
                "avg_input_length": 0.0,
                "avg_output_length": 0.0,
                "per_task": {},
                "per_document": {},
            }

        input_lengths = [len(ex.input_text) for ex in dataset.examples]
        output_lengths = [len(ex.output_text) for ex in dataset.examples]

        # Per-task stats
        per_task: Dict[str, Dict[str, Any]] = {}
        for ex in dataset.examples:
            tname = ex.task_name
            stats = per_task.setdefault(
                tname,
                {
                    "count": 0,
                    "total_input_len": 0,
                    "total_output_len": 0,
                },
            )
            stats["count"] += 1
            stats["total_input_len"] += len(ex.input_text)
            stats["total_output_len"] += len(ex.output_text)

        for _tname, stats in per_task.items():
            c = stats["count"]
            stats["avg_input_length"] = stats["total_input_len"] / c
            stats["avg_output_length"] = stats["total_output_len"] / c

        # Per-document stats
        per_document: Dict[str, Dict[str, Any]] = {}
        for ex in dataset.examples:
            doc_id = ex.document_id
            stats = per_document.setdefault(
                doc_id,
                {
                    "count": 0,
                },
            )
            stats["count"] += 1

        return {
            "num_examples": num_examples,
            "avg_input_length": sum(input_lengths) / num_examples,
            "avg_output_length": sum(output_lengths) / num_examples,
            "per_task": per_task,
            "per_document": per_document,
        }

    async def export_dataset(
        self,
        dataset_id: str,
        output_path: Union[str, Path],
    ) -> Path:
        dataset = self.datasets.get(dataset_id)
        if dataset is None:
            raise TrainingDataBotError(f"Unknown dataset_id: {dataset_id!r}")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            for ex in dataset.examples:
                record = {
                    "id": ex.id,
                    "task_name": ex.task_name,
                    "task_type": ex.task_type.value,
                    "input_text": ex.input_text,
                    "output_text": ex.output_text,
                    "document_id": ex.document_id,
                    "chunk_id": ex.chunk_id,
                    "model_name": ex.model_name,
                    "task_version": ex.task_version,
                    "created_at": ex.created_at.isoformat() if ex.created_at else None,
                    "temperature": ex.temperature,
                    "metadata": ex.metadata,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self.logger.info("Exported dataset %s to %s", dataset_id, str(path))
        return path

    async def run_pipeline(
        self,
        sources: Iterable[Union[str, Path]],
        output_path: Union[str, Path],
        tasks: Optional[List[TaskTemplate]] = None,
        max_examples: Optional[int] = None,
        max_chars: int = 800,
        overlap: int = 100,
    ) -> Path:
        """
        One-click pipeline:
          1) load documents
          2) chunk documents
          3) process them
          4) export dataset
        """
        await self.load_documents(sources)
        await self.chunk_documents(max_chars=max_chars, overlap=overlap)
        dataset_id = await self.process_documents(
            tasks=tasks,
            max_examples=max_examples,
        )
        return await self.export_dataset(dataset_id, output_path)
