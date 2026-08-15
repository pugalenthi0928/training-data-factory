"""Forge's canonical typed pipeline API.

The historical ``training_data_robo`` package remains available while callers
migrate. New orchestration code should import from ``forge``.
"""

from .contracts import (
    ArtifactBinding,
    ArtifactRef,
    ContaminationConfig,
    DedupeConfig,
    DifficultyConfig,
    EvaluationConfig,
    GenerationConfig,
    IngestConfig,
    JudgmentConfig,
    ModelRef,
    PromptRef,
    QualityConfig,
    SelectionConfig,
    SplitConfig,
    TrainingConfig,
)
from .pipeline import Pipeline, StageContext, StageDefinition, StageExecutionError, StageResult
from .workflow import ForgeConfig, ForgeRun, build_pipeline, run_forge

__version__ = "0.10.0"

__all__ = [
    "ArtifactBinding",
    "ArtifactRef",
    "ContaminationConfig",
    "DedupeConfig",
    "DifficultyConfig",
    "EvaluationConfig",
    "ForgeConfig",
    "ForgeRun",
    "GenerationConfig",
    "IngestConfig",
    "JudgmentConfig",
    "ModelRef",
    "Pipeline",
    "PromptRef",
    "QualityConfig",
    "SelectionConfig",
    "SplitConfig",
    "StageContext",
    "StageDefinition",
    "StageExecutionError",
    "StageResult",
    "TrainingConfig",
    "build_pipeline",
    "run_forge",
]
