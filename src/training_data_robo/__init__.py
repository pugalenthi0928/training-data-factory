"""
🏭 Training Data Robo Factory

Enterprise-style system for preparing high-quality training datasets for LLMs.
"""

__title__ = "training_data_robo"
__description__ = "Enterprise-style training data factory for LLMs"
__version__ = "0.1.0"
__author__ = "Pugalenthi Magendran"

from .settings import Settings
from .logging_config import get_logger
from .errors import TrainingDataBotError
from .bot import TrainingDataBot
from .models import (
    Document,
    TextChunk,
    TaskType,
    TaskTemplate,
    TrainingExample,
    Dataset,
    ProcessingJob,
)

__all__ = [
    "Settings",
    "get_logger",
    "TrainingDataBotError",
    "TrainingDataBot",
    "Document",
    "TextChunk",
    "TaskType",
    "TaskTemplate",
    "TrainingExample",
    "Dataset",
    "ProcessingJob",
]
