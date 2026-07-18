from src.retrieval.text_normalization import (
    make_theme_search_document,
    normalize_text,
    tokenize_ru,
)
from src.retrieval.theme_schema import ThemeRecord


def test_normalization():
    assert normalize_text("  Ёлка\xa0  ЖКХ / ТСЖ ") == "елка жкх тсж"


def test_russian_tokenization():
    tokens = tokenize_ru("Перебои с отоплением")
    assert tokens
    assert all(token == token.lower() for token in tokens)


def test_search_document_has_context():
    theme = ThemeRecord(
        code="0005.0005.0056.1156",
        name="Перебои в теплоснабжении",
        section="0005",
        sectionName="ЖКХ",
        parentName="Коммунальное хозяйство",
        path=["ЖКХ", "Жилище", "Перебои в теплоснабжении"],
    )
    document = make_theme_search_document(theme)
    assert "перебои в теплоснабжении" in document
    assert "коммунальное хозяйство" in document
    assert "жкх" in document
    assert "жилище" in document
