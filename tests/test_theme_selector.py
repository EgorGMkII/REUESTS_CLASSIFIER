from src.classification.theme_selector import ThemeSelector
from src.retrieval.theme_schema import ThemeCandidate


def candidate(code: str, name: str, section: str = "0005") -> ThemeCandidate:
    return ThemeCandidate(
        code=code,
        name=name,
        section=section,
        path=[name],
        pathCodes=[code],
        hybridScore=0.9,
        rank=1,
        source="hybrid",
    )


def test_selector_trusts_only_candidate_metadata():
    candidates = [candidate("a", "Официальное имя")]
    result = ThemeSelector(
        lambda _: '{"themes":[{"code":"a","name":"Подмена","confidence":0.8}]}'
    ).select("текст", candidates)
    assert result[0].name == "Официальное имя"
    assert result[0].section == "0005"


def test_selector_ignores_unknowns_and_duplicates_and_limits():
    candidates = [candidate("a", "A"), candidate("b", "B"), candidate("c", "C")]
    response = """{"themes":[
        {"code":"outside","confidence":1},
        {"code":"a","confidence":0.9},
        {"code":"a","confidence":0.8},
        {"code":"b","confidence":0.7},
        {"code":"c","confidence":0.6}
    ]}"""
    result = ThemeSelector(lambda _: response).select(
        "текст", candidates, max_themes=2
    )
    assert [item.code for item in result] == ["a", "b"]


def test_selector_keeps_multiple_themes_for_multitheme_text():
    candidates = [candidate("a", "A"), candidate("b", "B"), candidate("c", "C")]
    response = """{"themes":[
        {"code":"a","confidence":0.9},
        {"code":"b","confidence":0.7},
        {"code":"c","confidence":0.6}
    ]}"""
    result = ThemeSelector(lambda _: response).select(
        "Прошу решить вопрос отопления. Также прошу убрать свалку.", candidates, max_themes=2
    )
    assert [item.code for item in result] == ["a", "b"]


def test_empty_llm_selection_falls_back_to_top_one():
    candidates = [candidate("a", "A"), candidate("b", "B")]
    result = ThemeSelector(lambda _: '{"themes":[]}').select("текст", candidates)
    assert len(result) == 1
    assert result[0].code == "a"
    assert result[0].confidence == 0.4


def test_selector_prompt_discourages_adjacent_themes():
    prompts = []
    candidates = [candidate("a", "A")]
    ThemeSelector(
        lambda prompt: prompts.append(prompt) or '{"themes":[{"code":"a","confidence":0.8}]}'
    ).select("текст", candidates, max_themes=3)
    assert "Верни от 1 до 3 тематик" in prompts[0]
    assert "не стремись искусственно сжать обращение до одной темы" in prompts[0]
    assert "Не добавляй соседние, родственные или более общие темы" in prompts[0]
    assert "Не выбирай узкую тему только потому, что она узкая" in prompts[0]
    assert "Категориальные темы выбирай только при прямом совпадении категории" in prompts[0]
    assert "инвалидность здесь может быть обоснованием нуждаемости" in prompts[0]
    assert "Фермеры сообщают о массовом забое коров" in prompts[0]


def test_no_candidates_returns_empty_without_llm():
    calls = []
    assert ThemeSelector(lambda prompt: calls.append(prompt)).select("x", []) == []
    assert calls == []
