import pytest

from src.classification.question_pair_selector import (
    QuestionPairCandidate,
    QuestionPairSelector,
)
from src.classification.schemas import QuestionSubtypePrediction, QuestionTypePrediction


def pair(type_code: str, subtype_code: str) -> QuestionPairCandidate:
    return QuestionPairCandidate(
        question_type=QuestionTypePrediction(
            code=type_code, name=f"type {type_code}", confidence=0.8
        ),
        question_subtype=QuestionSubtypePrediction(
            code=subtype_code,
            officialCode=f"official {subtype_code}",
            name=f"subtype {subtype_code}",
            confidence=0.7,
        ),
    )


def test_pair_selector_accepts_only_provided_pair():
    candidates = [pair("3", "3.2.1"), pair("2", "2.1.1")]
    result = QuestionPairSelector(lambda _: '{"key":"2|2.1.1","confidence":0.8}').select(
        "текст", candidates
    )
    assert result.key == "2|2.1.1"


def test_pair_selector_falls_back_to_first_pair_on_invalid_response():
    candidates = [pair("3", "3.2.1"), pair("2", "2.1.1")]
    result = QuestionPairSelector(lambda _: '{"key":"9|9.9","confidence":0.8}').select(
        "текст", candidates
    )
    assert result.key == "3|3.2.1"


def test_pair_selector_does_not_call_llm_for_single_candidate():
    calls = []
    result = QuestionPairSelector(lambda prompt: calls.append(prompt)).select(
        "текст", [pair("2", "2.1.1")]
    )
    assert result.key == "2|2.1.1"
    assert calls == []


def test_pair_selector_rejects_empty_candidates():
    with pytest.raises(ValueError):
        QuestionPairSelector().select("текст", [])


def test_pair_selector_prompt_contains_statement_vs_complaint_logic():
    prompts = []
    QuestionPairSelector(
        lambda prompt: prompts.append(prompt)
        or '{"key":"2|2.1.1","confidence":0.8}'
    ).select("текст", [pair("3", "3.2.1"), pair("2", "2.1.1")])
    assert "Не выбирай \"Жалоба\" только потому" in prompts[0]
    assert "провести проверку" in prompts[0]
    assert "Выбирай пару целиком" in prompts[0]
    assert "если среди кандидатов есть и 2 Заявление, и 3 Жалоба" in prompts[0]
    assert "Жалобный стиль не равен виду 3" in prompts[0]
    assert "Массовая проблема" in prompts[0]
    assert "оценить ущерб или компенсацию" in prompts[0]


def test_pair_selector_prompt_contains_statement_vs_information_request_logic():
    prompts = []
    QuestionPairSelector(
        lambda prompt: prompts.append(prompt)
        or '{"key":"2|2.1.1","confidence":0.8}'
    ).select("текст", [pair("5", "-"), pair("2", "2.1.1")])
    assert "если среди кандидатов есть и 2 Заявление, и 5 Запрос информации" in prompts[0]
    assert "сведения нужны как часть решения" in prompts[0]
    assert "не делают вид 5 автоматически" in prompts[0]
