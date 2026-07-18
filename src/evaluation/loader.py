from __future__ import annotations

import json
from pathlib import Path

from src.classification.question_type_classifier import QUESTION_TYPES
from src.classification.subtype_catalog import QuestionSubtypeCatalog
from src.retrieval.theme_schema import ThemeRecord

from .schemas import EvaluationDataset


def load_evaluation_dataset(
    path: Path,
    themes: list[ThemeRecord],
    subtype_catalog: QuestionSubtypeCatalog,
) -> EvaluationDataset:
    dataset = EvaluationDataset.model_validate_json(path.read_text(encoding="utf-8"))
    if dataset.classifierVersion != subtype_catalog.catalogVersion:
        raise ValueError("Dataset and subtype catalog versions do not match")
    if any(theme.classifierVersion != dataset.classifierVersion for theme in themes):
        raise ValueError("Dataset and theme classifier versions do not match")
    ids = [case.id for case in dataset.cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation dataset contains duplicate case IDs")
    themes_by_code = {theme.code: theme for theme in themes}
    subtypes = subtype_catalog.by_code()
    for case in dataset.cases:
        expected = case.expected
        official_type_name = QUESTION_TYPES.get(expected.questionType.code)
        if official_type_name != expected.questionType.name:
            raise ValueError(f"{case.id}: invalid question type")
        subtype = subtypes.get(expected.questionSubtype.code)
        if subtype is None:
            raise ValueError(f"{case.id}: unknown subtype code")
        if expected.questionType.code not in subtype.allowedQuestionTypes:
            raise ValueError(f"{case.id}: subtype is not allowed for question type")
        if (
            subtype.name != expected.questionSubtype.name
            or subtype.officialCode != expected.questionSubtype.officialCode
        ):
            raise ValueError(f"{case.id}: subtype metadata does not match catalog")
        for expected_theme in expected.themes:
            theme = themes_by_code.get(expected_theme.code)
            if theme is None:
                raise ValueError(f"{case.id}: unknown theme code {expected_theme.code}")
            if theme.name != expected_theme.name or theme.section != expected_theme.section:
                raise ValueError(f"{case.id}: theme metadata does not match catalog")
    return dataset
