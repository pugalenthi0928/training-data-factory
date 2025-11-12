from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class DocumentSource(str, Enum):
    TEXT = "text"
    FILE = "file"
    URL = "url"


@dataclass
class Document:
    id: str
    title: str
    content: str
    source: DocumentSource
    path: Optional[str] = None
    url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(
        cls,
        path: str,
        content: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Document":
        from uuid import uuid4

        return cls(
            id=str(uuid4()),
            title=title or path,
            content=content,
            source=DocumentSource.FILE,
            path=path,
            metadata=metadata or {},
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        content: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Document":
        from uuid import uuid4

        return cls(
            id=str(uuid4()),
            title=title or url,
            content=content,
            source=DocumentSource.URL,
            url=url,
            metadata=metadata or {},
        )

    @classmethod
    def from_text(
        cls,
        content: str,
        title: str = "Untitled",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Document":
        from uuid import uuid4

        return cls(
            id=str(uuid4()),
            title=title,
            content=content,
            source=DocumentSource.TEXT,
            metadata=metadata or {},
        )


@dataclass
class TextChunk:
    id: str
    document_id: str
    index: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_document(
        cls,
        document: Document,
        text: str,
        index: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "TextChunk":
        from uuid import uuid4

        return cls(
            id=str(uuid4()),
            document_id=document.id,
            index=index,
            text=text,
            metadata=metadata or {},
        )


class TaskType(str, Enum):
    QA = "qa"
    SUMMARISATION = "summarisation"
    KEY_POINTS = "key_points"
    TITLE = "title"
    CLASSIFICATION = "classification"


@dataclass
class TaskTemplate:
    name: str
    task_type: TaskType
    system_prompt: str
    user_prompt_template: str
    max_output_tokens: int = 512

    # Metadata / knobs
    version: str = "v1"
    temperature: float = 0.2
    top_p: float = 1.0


@dataclass
class TrainingExample:
    id: str
    task_name: str
    task_type: TaskType
    input_text: str
    output_text: str
    document_id: str
    chunk_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # New metadata fields
    model_name: str = "unknown"
    task_version: str = "v1"
    created_at: datetime = field(default_factory=datetime.utcnow)
    temperature: float = 0.2


@dataclass
class Dataset:
    id: str
    name: str
    created_at: datetime
    examples: List[TrainingExample] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingJob:
    id: str
    status: ProcessingStatus
    created_at: datetime
    updated_at: datetime
    document_ids: List[str]
    task_names: List[str]
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def new_dataset(name: str, metadata: Optional[Dict[str, Any]] = None) -> Dataset:
    from uuid import uuid4

    return Dataset(
        id=str(uuid4()),
        name=name,
        created_at=datetime.utcnow(),
        metadata=metadata or {},
    )


def new_job(
    document_ids: List[str],
    task_names: List[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> ProcessingJob:
    from uuid import uuid4

    now = datetime.utcnow()
    return ProcessingJob(
        id=str(uuid4()),
        status=ProcessingStatus.PENDING,
        created_at=now,
        updated_at=now,
        document_ids=document_ids,
        task_names=task_names,
        metadata=metadata or {},
    )
