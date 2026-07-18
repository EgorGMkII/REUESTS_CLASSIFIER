from .question_subtype_classifier import ALLOWED_SUBTYPES
from .question_type_classifier import QUESTION_TYPES
from .schemas import (
    ClassificationResult,
    QuestionSubtypePrediction,
    QuestionTypePrediction,
    ThemePrediction,
)


def validate_themes(
    themes: list[ThemePrediction], allowed_theme_codes: set[str]
) -> list[str]:
    errors: list[str] = []
    if not themes:
        errors.append("themes must not be empty")
    seen: set[str] = set()
    for theme in themes:
        if theme.code not in allowed_theme_codes:
            errors.append(f"unknown theme code: {theme.code}")
        if theme.code in seen:
            errors.append(f"duplicate theme code: {theme.code}")
        seen.add(theme.code)
        if not 0.0 <= theme.confidence <= 1.0:
            errors.append(f"invalid theme confidence: {theme.code}")
    return errors


def validate_question_type(question_type: QuestionTypePrediction) -> list[str]:
    errors: list[str] = []
    official_name = QUESTION_TYPES.get(question_type.code)
    if official_name is None:
        errors.append(f"unknown question type: {question_type.code}")
    elif question_type.name != official_name:
        errors.append(f"invalid question type name: {question_type.name}")
    if not 0.0 <= question_type.confidence <= 1.0:
        errors.append("invalid question type confidence")
    return errors


def validate_question_subtype(
    question_type: QuestionTypePrediction,
    question_subtype: QuestionSubtypePrediction,
) -> list[str]:
    errors: list[str] = []
    allowed = ALLOWED_SUBTYPES.get(question_type.code)
    if allowed is None:
        errors.append(f"no subtype classifier for question type: {question_type.code}")
    elif question_subtype.code not in allowed:
        errors.append(
            f"subtype {question_subtype.code} is not allowed for {question_type.code}"
        )
    if question_type.code in {"5", "6", "7", "8", "9"} and question_subtype.code != "-":
        errors.append(f"question type {question_type.code} requires subtype -")
    from .question_subtype_classifier import _DEFAULT_CATALOG

    definition = _DEFAULT_CATALOG.by_code().get(question_subtype.code)
    if definition is not None:
        if question_subtype.name != definition.name:
            errors.append(f"invalid question subtype name: {question_subtype.name}")
        if question_subtype.officialCode != definition.officialCode:
            errors.append(
                f"invalid question subtype official code: {question_subtype.officialCode}"
            )
    if not 0.0 <= question_subtype.confidence <= 1.0:
        errors.append("invalid question subtype confidence")
    return errors


def validate_classification_result(
    result: ClassificationResult, allowed_theme_codes: set[str]
) -> list[str]:
    return [
        *validate_themes(result.themes, allowed_theme_codes),
        *validate_question_type(result.questionType),
        *validate_question_subtype(result.questionType, result.questionSubtype),
    ]
