"""Sequential classification pipeline for citizen requests."""

from .errors import ClassificationError, ClassificationValidationError
from .schemas import (
    ClassificationMeta,
    ClassificationResult,
    QuestionSubtypePrediction,
    QuestionTypePrediction,
    ThemePrediction,
)

__all__ = [
    "ClassificationError",
    "ClassificationMeta",
    "ClassificationPipeline",
    "ClassificationResult",
    "ClassificationValidationError",
    "QuestionSubtypePrediction",
    "QuestionTypePrediction",
    "ThemePrediction",
]


def __getattr__(name: str):
    if name == "ClassificationPipeline":
        from .pipeline import ClassificationPipeline

        return ClassificationPipeline
    raise AttributeError(name)
