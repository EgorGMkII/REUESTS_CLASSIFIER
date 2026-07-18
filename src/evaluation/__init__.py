"""Offline quality evaluation for the classification pipeline."""

from .loader import load_evaluation_dataset
from .runner import EvaluationRunner

__all__ = ["EvaluationRunner", "load_evaluation_dataset"]
