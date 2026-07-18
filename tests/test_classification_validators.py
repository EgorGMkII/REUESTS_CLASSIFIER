from src.classification.schemas import (
    ClassificationMeta,
    ClassificationResult,
    QuestionSubtypePrediction,
    QuestionTypePrediction,
    ThemePrediction,
)
from src.classification.validators import validate_classification_result


def result(
    themes=None,
    question_type=None,
    subtype=None,
) -> ClassificationResult:
    return ClassificationResult(
        themes=themes
        if themes is not None
        else [ThemePrediction(code="theme", name="Theme", confidence=0.8, section="1")],
        questionType=question_type
        or QuestionTypePrediction(code="3", name="Жалоба", confidence=0.8),
        questionSubtype=subtype
        or QuestionSubtypePrediction(
            code="3.1.1",
            officialCode="Ж1.1",
            name="Просьба автора обращения о восстановлении или защите нарушенных его прав",
            confidence=0.7,
        ),
        meta=ClassificationMeta(
            requestId="id", processedAt="2026-07-03T00:00:00Z", sourceTextLength=1
        ),
    )


def test_valid_result():
    assert validate_classification_result(result(), {"theme"}) == []


def test_empty_unknown_and_duplicate_themes():
    assert validate_classification_result(result(themes=[]), {"theme"})
    unknown = [ThemePrediction(code="bad", name="X", confidence=1, section="1")]
    assert any("unknown theme" in error for error in validate_classification_result(result(themes=unknown), {"theme"}))
    duplicate = [
        ThemePrediction(code="theme", name="A", confidence=1, section="1"),
        ThemePrediction(code="theme", name="A", confidence=1, section="1"),
    ]
    assert any("duplicate" in error for error in validate_classification_result(result(themes=duplicate), {"theme"}))


def test_invalid_question_type_code_and_name():
    invalid_code = QuestionTypePrediction(code="0", name="X", confidence=1)
    assert any("unknown question type" in error for error in validate_classification_result(result(question_type=invalid_code), {"theme"}))
    invalid_name = QuestionTypePrediction(code="3", name="Заявление", confidence=1)
    assert any("invalid question type name" in error for error in validate_classification_result(result(question_type=invalid_name), {"theme"}))


def test_invalid_subtype_and_rule_for_types_five_to_nine():
    bad = QuestionSubtypePrediction(
        code="1.1.1",
        officialCode="П1.1",
        name="Рекомендации автора обращения по совершенствованию законов",
        confidence=1,
    )
    assert any("not allowed" in error for error in validate_classification_result(result(subtype=bad), {"theme"}))
    info = QuestionTypePrediction(code="5", name="Запрос информации", confidence=1)
    assert any("requires subtype" in error for error in validate_classification_result(result(question_type=info, subtype=bad), {"theme"}))
