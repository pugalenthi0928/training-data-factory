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
    ProfileConfig,
    PromptRef,
    QualityConfig,
    RecordGovernanceConfig,
    SelectionConfig,
    SourceGovernanceConfig,
    SplitConfig,
    TrainingConfig,
)
from .evaluation import (
    EvaluationValidationError,
    analyse_evaluation,
    cohen_kappa,
    freeze_evaluation_set,
    krippendorff_alpha_nominal,
    verify_evaluation_release,
)
from .pairwise_judge import PAIRWISE_JUDGE_PROMPT_SHA256, PAIRWISE_JUDGE_PROMPT_VERSION
from .pipeline import Pipeline, StageContext, StageDefinition, StageExecutionError, StageResult
from .workflow import ForgeConfig, ForgeRun, build_pipeline, run_forge

__version__ = "0.13.0"

__all__ = [
    "ArtifactBinding",
    "ArtifactRef",
    "ContaminationConfig",
    "DedupeConfig",
    "DifficultyConfig",
    "EvaluationConfig",
    "EvaluationValidationError",
    "ForgeConfig",
    "ForgeRun",
    "GenerationConfig",
    "IngestConfig",
    "JudgmentConfig",
    "ModelRef",
    "PAIRWISE_JUDGE_PROMPT_SHA256",
    "PAIRWISE_JUDGE_PROMPT_VERSION",
    "Pipeline",
    "PromptRef",
    "ProfileConfig",
    "QualityConfig",
    "RecordGovernanceConfig",
    "SelectionConfig",
    "SourceGovernanceConfig",
    "SplitConfig",
    "StageContext",
    "StageDefinition",
    "StageExecutionError",
    "StageResult",
    "TrainingConfig",
    "analyse_evaluation",
    "build_pipeline",
    "cohen_kappa",
    "freeze_evaluation_set",
    "krippendorff_alpha_nominal",
    "run_forge",
    "verify_evaluation_release",
]
