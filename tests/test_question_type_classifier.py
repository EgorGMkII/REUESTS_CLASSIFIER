from src.classification.question_type_classifier import QuestionTypeClassifier


def test_valid_question_type_uses_official_name():
    result = QuestionTypeClassifier(
        lambda _: '{"code":"3","name":"Подмена","confidence":0.82}'
    ).classify("жалоба")
    assert (result.code, result.name, result.confidence) == ("3", "Жалоба", 0.82)


def test_unknown_question_type_falls_back():
    result = QuestionTypeClassifier(
        lambda _: '{"code":"42","confidence":0.9}'
    ).classify("текст")
    assert (result.code, result.name, result.confidence) == ("2", "Заявление", 0.3)


def test_invalid_json_falls_back():
    result = QuestionTypeClassifier(lambda _: "not json").classify("текст")
    assert result.code == "2"
    assert result.confidence == 0.3


def test_top_k_question_types_accepts_valid_candidates_and_deduplicates():
    result = QuestionTypeClassifier(
        lambda _: """
        {"candidates": [
          {"code":"3","confidence":0.55},
          {"code":"2","confidence":0.40},
          {"code":"3","confidence":0.30}
        ]}
        """
    ).classify_top_k("спорный текст", k=3)
    assert [(item.code, item.name, item.confidence) for item in result] == [
        ("3", "Жалоба", 0.55),
        ("2", "Заявление", 0.4),
    ]
    assert result[0].reason == ""


def test_top_k_adds_statement_alternative_for_uncertain_complaint_or_info_request():
    result = QuestionTypeClassifier(
        lambda _: """
        {"candidates": [
          {"code":"5","confidence":0.70,"reason":"просит сообщить сведения"}
        ]}
        """
    ).classify_top_k("спорный запрос сведений для решения проблемы", k=2)
    assert [(item.code, item.name) for item in result] == [
        ("5", "Запрос информации"),
        ("2", "Заявление"),
    ]
    assert result[0].reason == "просит сообщить сведения"
    assert "альтернатива" in result[1].reason


def test_top_k_question_types_falls_back_when_no_valid_candidates():
    result = QuestionTypeClassifier(
        lambda _: '{"candidates": [{"code":"42","confidence":0.9}]}'
    ).classify_top_k("текст")
    assert len(result) == 1
    assert result[0].code == "2"
    assert result[0].confidence == 0.3


def test_top_k_prompt_contains_statement_vs_complaint_logic():
    prompts = []
    QuestionTypeClassifier(
        lambda prompt: prompts.append(prompt)
        or '{"candidates": [{"code":"2","confidence":0.7}]}'
    ).classify_top_k("текст", k=2)
    assert "проведения проверки" in prompts[0]
    assert "Не выбирай жалобу только по словам" in prompts[0]
    assert "2 Заявление, и 3 Жалоба" in prompts[0]
    assert "Слова \"нарушение\", \"незаконно\", \"бездействие\"" in prompts[0]
    assert "Граница 5 Запрос информации vs 2 Заявление" in prompts[0]
    assert "Если автор просит сведения, чтобы решить конкретную жизненную проблему" in prompts[0]
    assert "Обычное обращение гражданина почти никогда не является видом 6" in prompts[0]
