from pathlib import Path
from unittest.mock import Mock

import pytest

from src.classification.errors import ClassificationValidationError
from src.classification.pipeline import ClassificationPipeline, build_pipeline
from src.classification.question_pair_selector import QuestionPairCandidate
from src.classification.question_type_classifier import QUESTION_TYPES
from src.classification.schemas import (
    QuestionSubtypePrediction,
    QuestionTypePrediction,
    ThemePrediction,
)
from src.classification.subtype_catalog import load_subtype_catalog
from src.classification.text_preprocessor import LLMTextPreprocessor
from src.classification.text_preprocessor import RetrievalQuery
from src.retrieval.theme_schema import ThemeCandidate


def candidate(
    code: str = "theme",
    section: str = "0005",
    score: float = 0.9,
    rank: int = 1,
) -> ThemeCandidate:
    return ThemeCandidate(
        code=code,
        name=f"Theme {code}",
        section=section,
        path=[f"Theme {code}"],
        pathCodes=[code],
        hybridScore=score,
        rank=rank,
        source="hybrid",
    )


def components(selected=None, candidates=None):
    retriever = Mock()
    retriever.search.return_value = [candidate()] if candidates is None else candidates
    type_classifier = Mock()
    type_prediction = QuestionTypePrediction(
        code="3", name="Жалоба", confidence=0.8
    )
    type_classifier.classify_top_k.return_value = [type_prediction]
    subtype_classifier = Mock()
    subtype_prediction = QuestionSubtypePrediction(
        code="3.1.1",
        officialCode="Ж1.1",
        name="Просьба автора обращения о восстановлении или защите нарушенных его прав",
        confidence=0.7,
    )
    subtype_classifier.classify.return_value = subtype_prediction
    subtype_classifier.classify_top_k.return_value = [subtype_prediction]
    pair_selector = Mock()
    pair_selector.select.return_value = QuestionPairCandidate(
        question_type=type_prediction,
        question_subtype=subtype_prediction,
    )
    selector = Mock()
    selector.select.return_value = (
        [ThemePrediction(code="theme", name="Theme", confidence=0.9, section="0005")]
        if selected is None
        else selected
    )
    pipeline = ClassificationPipeline(
        retriever,
        type_classifier,
        subtype_classifier,
        pair_selector,
        selector,
        {"theme"},
    )
    return pipeline, retriever, type_classifier, subtype_classifier, pair_selector, selector


def test_pipeline_coordinates_components_and_builds_result():
    pipeline, retriever, type_classifier, subtype_classifier, pair_selector, selector = (
        components()
    )
    result = pipeline.run("  Жалоба Ёлка  ")
    retriever.search.assert_called_once_with("жалоба елка", top_k=30)
    type_classifier.classify_top_k.assert_called_once_with("жалоба елка", k=2)
    subtype_classifier.classify_top_k.assert_called_once_with(
        "жалоба елка", type_classifier.classify_top_k.return_value[0], k=2
    )
    pair_selector.select.assert_called_once()
    assert pair_selector.select.call_args.args[0] == "жалоба елка"
    assert pair_selector.select.call_args.args[1][0].question_type.code == "3"
    assert pair_selector.select.call_args.args[1][0].question_subtype.code == "3.1.1"
    selector.select.assert_called_once_with(
        "жалоба елка", pipeline.last_candidates, max_themes=3
    )
    assert result.themes[0].code == "theme"
    assert result.meta.sourceTextLength == len("  Жалоба Ёлка  ")
    assert len(result.meta.requestId) == 32


def test_pipeline_uses_text_preprocessor_before_normalization():
    preprocessor = Mock()
    preprocessor.prepare.return_value = "Очищенная жалоба про Ёлку"
    pipeline, retriever, *_ = components()
    pipeline.text_preprocessor = preprocessor
    pipeline.run("сырой OCR")
    preprocessor.prepare.assert_called_once_with("сырой OCR")
    retriever.search.assert_called_once_with("очищенная жалоба про елку", top_k=30)
    assert pipeline.last_preprocessed_text == "Очищенная жалоба про Ёлку"


def test_pipeline_builds_subtype_for_each_type_candidate_and_uses_selected_pair():
    pipeline, _, type_classifier, subtype_classifier, pair_selector, _ = components()
    type_candidates = [
        QuestionTypePrediction(code="3", name="Жалоба", confidence=0.6),
        QuestionTypePrediction(code="2", name="Заявление", confidence=0.5),
    ]
    subtype_candidates = [
        QuestionSubtypePrediction(
            code="3.2.1",
            officialCode="Ж2.1",
            name="Жалоба на действия",
            confidence=0.7,
        ),
        QuestionSubtypePrediction(
            code="2.1.1",
            officialCode="З1.1",
            name="Просьба автора обращения о содействии в реализации его конституционных прав",
            confidence=0.8,
        ),
    ]
    type_classifier.classify_top_k.return_value = type_candidates
    subtype_classifier.classify_top_k.side_effect = [
        [subtype_candidates[0]],
        [subtype_candidates[1]],
    ]
    pair_selector.select.return_value = QuestionPairCandidate(
        question_type=type_candidates[1],
        question_subtype=subtype_candidates[1],
    )

    result = pipeline.run("Прошу принять меры")

    assert subtype_classifier.classify_top_k.call_count == 2
    assert [call.args[1].code for call in subtype_classifier.classify_top_k.call_args_list] == [
        "3",
        "2",
    ]
    assert [candidate.key for candidate in pipeline.last_question_pair_candidates] == [
        "3|3.2.1",
        "2|2.1.1",
    ]
    assert result.questionType.code == "2"
    assert result.questionSubtype.code == "2.1.1"


def test_pipeline_builds_two_subtype_candidates_per_type():
    pipeline, _, type_classifier, subtype_classifier, pair_selector, _ = components()
    type_candidate = QuestionTypePrediction(
        code="2", name=QUESTION_TYPES["2"], confidence=0.8
    )
    definitions = load_subtype_catalog().by_code()
    subtype_candidates = []
    for code, confidence in [("2.1.1", 0.7), ("2.2.1", 0.6)]:
        definition = definitions[code]
        subtype_candidates.append(
            QuestionSubtypePrediction(
                code=definition.code,
                officialCode=definition.officialCode,
                name=definition.name,
                confidence=confidence,
            )
        )
    type_classifier.classify_top_k.return_value = [type_candidate]
    subtype_classifier.classify_top_k.return_value = subtype_candidates
    pair_selector.select.return_value = QuestionPairCandidate(
        question_type=type_candidate,
        question_subtype=subtype_candidates[1],
    )

    result = pipeline.run("РїСЂРѕС€Сѓ РїСЂРѕРІРµСЂРёС‚СЊ Рё РїСЂРёРЅСЏС‚СЊ РјРµСЂС‹")

    subtype_classifier.classify_top_k.assert_called_once_with(
        pipeline.last_normalized_text,
        type_candidate,
        k=2,
    )
    assert [candidate.key for candidate in pipeline.last_question_pair_candidates] == [
        "2|2.1.1",
        "2|2.2.1",
    ]
    assert result.questionSubtype.code == "2.2.1"


def test_pipeline_skips_alternative_type_and_pair_selector_for_high_confidence_type():
    pipeline, _, type_classifier, subtype_classifier, pair_selector, _ = components()
    type_candidates = [
        QuestionTypePrediction(code="2", name=QUESTION_TYPES["2"], confidence=0.91),
        QuestionTypePrediction(code="3", name=QUESTION_TYPES["3"], confidence=0.5),
    ]
    definitions = load_subtype_catalog().by_code()
    subtype_candidates = []
    for code, confidence in [("2.1.1", 0.8), ("2.2.1", 0.6)]:
        definition = definitions[code]
        subtype_candidates.append(
            QuestionSubtypePrediction(
                code=definition.code,
                officialCode=definition.officialCode,
                name=definition.name,
                confidence=confidence,
            )
        )
    type_classifier.classify_top_k.return_value = type_candidates
    subtype_classifier.classify_top_k.return_value = subtype_candidates

    result = pipeline.run("РїСЂРѕС€Сѓ РїСЂРёРЅСЏС‚СЊ РјРµСЂС‹")

    subtype_classifier.classify_top_k.assert_called_once()
    assert subtype_classifier.classify_top_k.call_args.args[1].code == "2"
    pair_selector.select.assert_not_called()
    assert [candidate.key for candidate in pipeline.last_question_pair_candidates] == [
        "2|2.1.1",
        "2|2.2.1",
    ]
    assert result.questionType.code == "2"
    assert result.questionSubtype.code == "2.1.1"


def test_pipeline_uses_slow_path_when_second_type_is_plausible():
    pipeline, _, type_classifier, subtype_classifier, pair_selector, _ = components()
    type_candidates = [
        QuestionTypePrediction(code="3", name=QUESTION_TYPES["3"], confidence=0.93),
        QuestionTypePrediction(code="2", name=QUESTION_TYPES["2"], confidence=0.72),
    ]
    complaint_subtype = QuestionSubtypePrediction(
        code="3.2.1",
        officialCode="Ж2.1",
        name="Жалоба на действия",
        confidence=0.8,
    )
    definition = load_subtype_catalog().by_code()["2.1.1"]
    statement_subtype = QuestionSubtypePrediction(
        code=definition.code,
        officialCode=definition.officialCode,
        name=definition.name,
        confidence=0.7,
    )
    type_classifier.classify_top_k.return_value = type_candidates
    subtype_classifier.classify_top_k.side_effect = [
        [complaint_subtype],
        [statement_subtype],
    ]
    pair_selector.select.return_value = QuestionPairCandidate(
        question_type=type_candidates[1],
        question_subtype=statement_subtype,
    )

    result = pipeline.run("Прошу проверить и принять меры")

    assert subtype_classifier.classify_top_k.call_count == 2
    pair_selector.select.assert_called_once()
    assert [candidate.key for candidate in pipeline.last_question_pair_candidates] == [
        "3|3.2.1",
        "2|2.1.1",
    ]
    assert result.questionType.code == "2"


def test_pipeline_uses_slow_path_for_information_request_with_statement_alternative():
    pipeline, _, type_classifier, subtype_classifier, pair_selector, _ = components()
    type_candidates = [
        QuestionTypePrediction(code="5", name=QUESTION_TYPES["5"], confidence=0.97),
        QuestionTypePrediction(code="2", name=QUESTION_TYPES["2"], confidence=0.19),
    ]
    info_subtype = QuestionSubtypePrediction(
        code="-",
        officialCode="-",
        name="Запрос информации",
        confidence=0.8,
    )
    definition = load_subtype_catalog().by_code()["2.1.1"]
    statement_subtype = QuestionSubtypePrediction(
        code=definition.code,
        officialCode=definition.officialCode,
        name=definition.name,
        confidence=0.6,
    )
    type_classifier.classify_top_k.return_value = type_candidates
    subtype_classifier.classify_top_k.side_effect = [
        [info_subtype],
        [statement_subtype],
    ]
    pair_selector.select.return_value = QuestionPairCandidate(
        question_type=type_candidates[1],
        question_subtype=statement_subtype,
    )

    result = pipeline.run("Прошу сообщить сведения для решения жилищной проблемы")

    assert subtype_classifier.classify_top_k.call_count == 2
    pair_selector.select.assert_called_once()
    assert [candidate.key for candidate in pipeline.last_question_pair_candidates] == [
        "5|-",
        "2|2.1.1",
    ]
    assert result.questionType.code == "2"


def test_llm_text_preprocessor_returns_normalized_text():
    preprocessor = LLMTextPreprocessor(
        llm=lambda prompt: '{"normalizedText": "Прошу отремонтировать дорогу у дома."}'
    )
    assert preprocessor.prepare("OCR шум") == "Прошу отремонтировать дорогу у дома."
    assert preprocessor.last_retrieval_queries[0].query == "Прошу отремонтировать дорогу у дома."


def test_llm_text_preprocessor_returns_decision_text_and_retrieval_queries():
    preprocessor = LLMTextPreprocessor(
        llm=lambda prompt: """
        {
          "decisionText": "Автор просит разобраться с работой автобуса.",
          "typeDecisionText": "Обычное обращение гражданина: автор просит разобраться с работой автобуса и принять меры.",
          "anchorId": "passenger_transport",
          "mainSubject": "транспортное обслуживание населения",
          "domain": "пассажирские перевозки",
          "primaryProblem": "автобус не остановился",
          "secondaryProblems": ["остановочный пункт"],
          "detailsToDrop": ["номер автобуса"],
          "retrievalQueries": [
            {"section":"0003","query":"транспортное обслуживание населения пассажирские перевозки автобус маршрут остановка"},
            {"section":"9999","query":"этот запрос должен быть отброшен"},
            {"section":"0003","query":"общественный транспорт автобус не остановился остановочный пункт"},
            {"section":"0002","query":"пассажир получил социальную услугу"},
            {"section":"0005","query":"лишний запрос сверх лимита"},
            {"section":"0004","query":"еще один лишний запрос"}
          ]
        }
        """
    )
    assert preprocessor.prepare("OCR шум") == "Автор просит разобраться с работой автобуса."
    assert (
        preprocessor.last_type_decision_text
        == "Обычное обращение гражданина: автор просит разобраться с работой автобуса и принять меры."
    )
    assert preprocessor.last_retrieval_decomposition == {
        "anchorId": "passenger_transport",
        "mainSubject": "транспортное обслуживание населения",
        "domain": "пассажирские перевозки",
        "primaryProblem": "автобус не остановился",
        "secondaryProblems": ["остановочный пункт"],
        "detailsToDrop": ["номер автобуса"],
    }
    assert [(q.section, q.query) for q in preprocessor.last_retrieval_queries] == [
        (
            "0003",
            "транспортное обслуживание населения пассажирские перевозки автобус маршрут остановка",
        ),
        ("0003", "общественный транспорт автобус не остановился остановочный пункт"),
        ("0002", "пассажир получил социальную услугу"),
        ("0005", "лишний запрос сверх лимита"),
    ]


def test_llm_text_preprocessor_falls_back_on_invalid_response():
    preprocessor = LLMTextPreprocessor(llm=lambda prompt: "not json")
    assert preprocessor.prepare("  текст\n\n\nс OCR  ") == "текст\n\nс OCR"


def test_llm_text_preprocessor_prompt_asks_for_concise_key_terms():
    prompts = []
    preprocessor = LLMTextPreprocessor(
        llm=lambda prompt: prompts.append(prompt)
        or '{"normalizedText": "Прошу проверить свалку и принять меры."}'
    )
    preprocessor.prepare("OCR шум про свалку")
    assert "6–12 содержательных предложений" in prompts[0]
    assert "ключевые тематические слова" in prompts[0]
    assert "длинные цитаты законов" in prompts[0]
    assert "Компактный список тематических якорей" in prompts[0]
    assert "typeDecisionText" in prompts[0]
    assert "административный контекст" in prompts[0]
    assert "не называй итоговый вид обращения" in prompts[0]
    assert "Жалобный стиль" not in prompts[0]
    assert "mainSubject" in prompts[0]
    assert "эксплуатация и сохранность автомобильных дорог" in prompts[0]


def test_pipeline_uses_type_decision_text_for_question_type_and_subtype():
    preprocessor = Mock()
    preprocessor.prepare.return_value = "Короткий текст для тем"
    preprocessor.last_type_decision_text = (
        "Обычное обращение гражданина: автор просит содействие и принятие мер"
    )
    preprocessor.last_retrieval_queries = [
        RetrievalQuery(section="0005", query="жилищные условия")
    ]
    pipeline, _, type_classifier, subtype_classifier, pair_selector, _ = components()
    pipeline.text_preprocessor = preprocessor

    pipeline.run("сырой OCR")

    type_classifier.classify_top_k.assert_called_once_with(
        "обычное обращение гражданина автор просит содействие и принятие мер",
        k=2,
    )
    subtype_classifier.classify_top_k.assert_called_once()
    assert subtype_classifier.classify_top_k.call_args.args[0] == (
        "обычное обращение гражданина автор просит содействие и принятие мер"
    )
    pair_selector.select.assert_called_once()
    assert pair_selector.select.call_args.args[0] == (
        "обычное обращение гражданина автор просит содействие и принятие мер"
    )


def test_llm_text_preprocessor_works_without_anchor_file(tmp_path):
    preprocessor = LLMTextPreprocessor(
        llm=lambda prompt: '{"decisionText": "Автор жалуется на дорогу.", "retrievalQueries": [{"section":"0003","query":"эксплуатация автомобильной дороги"}]}',
        topic_anchors_path=tmp_path / "missing.json",
    )
    assert preprocessor.prepare("OCR шум") == "Автор жалуется на дорогу."
    assert preprocessor.last_retrieval_queries[0].query == "эксплуатация автомобильной дороги"


def test_pipeline_applies_theme_fallback():
    pipeline, *_ = components(selected=[])
    result = pipeline.run("текст")
    assert result.themes[0].code == "theme"
    assert result.themes[0].confidence == 0.4


def test_pipeline_raises_when_no_themes_or_candidates():
    pipeline, *_ = components(selected=[], candidates=[])
    with pytest.raises(ClassificationValidationError) as error:
        pipeline.run("текст")
    assert "themes must not be empty" in error.value.errors


def test_pipeline_uses_multi_query_retrieval_and_merges_candidates():
    preprocessor = Mock()
    preprocessor.prepare.return_value = "Автор просит разобраться с автобусом"
    preprocessor.last_retrieval_queries = [
        RetrievalQuery(section="0003", query="пассажирские перевозки автобус"),
        RetrievalQuery(section="0003", query="общественный транспорт остановка"),
        RetrievalQuery(section="0002", query="социальная услуга пассажиру"),
    ]
    retriever = Mock()
    repeated = candidate("transport", "0003", 0.70)
    social = candidate("social", "0002", 0.72)
    other = candidate("other", "0001", 0.95)
    retriever.search.side_effect = [
        [repeated],
        [repeated],
        [social, other],
    ]
    pipeline, _, type_classifier, subtype_classifier, pair_selector, selector = components()
    pipeline.text_preprocessor = preprocessor
    pipeline.theme_retriever = retriever

    result = pipeline.run("сырой OCR")

    assert result.themes
    assert [call.args[0] for call in retriever.search.call_args_list] == [
        "пассажирские перевозки автобус",
        "общественный транспорт остановка",
        "социальная услуга пассажиру",
    ]
    assert [call.kwargs["top_k"] for call in retriever.search.call_args_list] == [
        30,
        30,
        30,
    ]
    assert [candidate.code for candidate in pipeline.last_candidates] == [
        "other",
        "transport",
        "social",
    ]
    transport = next(item for item in pipeline.last_candidates if item.code == "transport")
    assert transport.hybridScore == pytest.approx(0.81)
    type_classifier.classify_top_k.assert_called_once_with(
        "автор просит разобраться с автобусом", k=2
    )
    subtype_classifier.classify_top_k.assert_called()
    pair_selector.select.assert_called()
    selector.select.assert_called_once()


def test_pipeline_limits_merged_theme_candidates_to_top_30():
    pipeline, retriever, *_ = components()
    retriever.search.return_value = [
        candidate(f"theme-{index:02d}", "0005", 1 - index / 100)
        for index in range(35)
    ]

    candidates = pipeline._search_theme_candidates(
        [RetrievalQuery(section="0005", query="жилищные условия")],
    )

    assert len(candidates) == 30
    assert candidates[0].code == "theme-00"
    assert candidates[-1].code == "theme-29"


def test_stale_real_artifacts_are_rejected():
    root = Path("data/evaluation/test-output")
    themes_path = root / "stale_themes.json"
    index_dir = root / "stale_index"
    index_dir.mkdir(exist_ok=True)
    themes_path.write_text(
        '[{"code":"theme","name":"Theme","section":"0005","isLeaf":true}]',
        encoding="utf-8",
    )
    (index_dir / "index_report.json").write_text(
        '{"themesSha256":"stale"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stale"):
        build_pipeline(
            themes_path,
            index_dir,
            Path("data/classifiers/question_subtypes.json"),
        )
    themes_path.unlink()
    (index_dir / "index_report.json").unlink()
    index_dir.rmdir()
