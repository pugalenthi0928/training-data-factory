"""
🏭 Training Data Robo Factory

Enterprise-style system for preparing high-quality training datasets for LLMs.
"""

__title__ = "training_data_robo"
__description__ = "Enterprise-style training data factory for LLMs"
__version__ = "0.1.0"
__author__ = "Pugalenthi Magendran"

from .bot import TrainingDataBot
from .errors import TrainingDataBotError
from .io import count_jsonl_rows, iter_jsonl, load_jsonl, write_jsonl
from .logging_config import get_logger
from .models import (
    ChunkType,
    Dataset,
    DifficultyLevel,
    Document,
    JudgeRubric,
    ProcessingJob,
    QualityDimension,
    TaskTemplate,
    TaskType,
    TextChunk,
    TrainingExample,
)
from .pipeline import Pipeline, PipelineError, StepResult
from .settings import Settings
from .tracker import ExperimentTracker, RunInfo

__all__ = [
    "Settings",
    "get_logger",
    "TrainingDataBotError",
    "load_jsonl",
    "write_jsonl",
    "iter_jsonl",
    "count_jsonl_rows",
    "TrainingDataBot",
    "Document",
    "TextChunk",
    "TaskType",
    "TaskTemplate",
    "TrainingExample",
    "Dataset",
    "ProcessingJob",
    "ChunkType",
    "DifficultyLevel",
    "QualityDimension",
    "JudgeRubric",
    "Pipeline",
    "PipelineError",
    "StepResult",
    "ExperimentTracker",
    "RunInfo",
]
