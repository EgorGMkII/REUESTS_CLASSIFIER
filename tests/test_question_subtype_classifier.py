import pytest

from src.classification.question_subtype_classifier import (
    QuestionSubtypeClassifier,
)
from src.classification.schemas import QuestionTypePrediction


def question_type(code: str) -> QuestionTypePrediction:
    return QuestionTypePrediction(code=code, name="test", confidence=1)


@pytest.mark.parametrize("code", ["5", "6", "7", "8", "9"])
def test_subtype_not_applicable_without_llm(code):
    calls = []
    result = QuestionSubtypeClassifier(lambda prompt: calls.append(prompt)).classify(
        "текст", question_type(code)
    )
    assert result.code == "-"
    assert result.confidence == 1
    assert calls == []


@pytest.mark.parametrize(
    ("type_code", "subtype_code"),
    [("1", "1.1.1"), ("2", "2.2.3"), ("3", "3.2.1"), ("4", "4.3")],
)
def test_allowed_subtype_is_accepted(type_code, subtype_code):
    result = QuestionSubtypeClassifier(
        lambda _: f'{{"code":"{subtype_code}","confidence":0.7}}'
    ).classify("текст", question_type(type_code))
    assert result.code == subtype_code
    assert result.officialCode != ""
    assert not result.name.startswith("Тип вида вопроса")


def test_subtype_prompt_contains_codes_official_codes_and_names():
    prompts = []
    QuestionSubtypeClassifier(
        lambda prompt: prompts.append(prompt) or '{"code":"3.1.1","confidence":0.7}'
    ).classify("текст", question_type("3"))
    assert "code=3.1.1" in prompts[0]
    assert "officialCode=Ж1.1" in prompts[0]
    assert "Просьба автора обращения о восстановлении или защите нарушенных его прав" in prompts[0]


def test_disallowed_subtype_falls_back_to_allowed_code():
    result = QuestionSubtypeClassifier(
        lambda _: '{"code":"9.9","confidence":0.8}'
    ).classify("текст", question_type("1"))
    assert result.code == "1.1.1"
    assert result.officialCode == "П1.1"
    assert result.confidence == 0.3
