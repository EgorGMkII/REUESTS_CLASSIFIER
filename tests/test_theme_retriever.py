import re
from pathlib import Path

import pytest

from src.retrieval.bm25_retriever import BM25ThemeRetriever
from src.retrieval.faiss_retriever import FaissThemeRetriever
from src.retrieval.hybrid_theme_retriever import HybridThemeRetriever
from src.retrieval.theme_schema import ThemeRecord


THEMES = [
    ThemeRecord(
        code="0005.0005.0056.1156",
        name="Перебои в теплоснабжении",
        section="0005",
        sectionName="ЖКХ",
        parentName="Коммунальное хозяйство",
        path=["ЖКХ", "Коммунальное хозяйство", "Перебои в теплоснабжении"],
    ),
    ThemeRecord(
        code="0004.0015.0155.0946",
        name="Строительство и ремонт дорог",
        section="0004",
        sectionName="Транспорт",
        path=["Транспорт", "Автомобильные дороги", "Строительство и ремонт дорог"],
    ),
    ThemeRecord(
        code="0001.0001.0001.0001",
        name="Конституционные права граждан",
        section="0001",
        sectionName="Государство",
        path=["Государство", "Конституционные права граждан"],
    ),
]


def embedding(text: str) -> list[float]:
    text = text.lower()
    return [
        float(len(re.findall(r"тепл|отоп", text))),
        float(len(re.findall(r"дорог|ремонт", text))),
        0.1,
    ]


def retrievers():
    bm25 = BM25ThemeRetriever(THEMES)
    vector = FaissThemeRetriever(THEMES, embedding_fn=embedding)
    vector.build()
    return bm25, vector, HybridThemeRetriever(bm25, vector)


def test_bm25_returns_relevant_theme():
    bm25, _, _ = retrievers()
    assert bm25.search("перебои отопления", 1)[0].section == "0005"


def test_vector_returns_relevant_theme():
    _, vector, _ = retrievers()
    assert vector.search("нужно отремонтировать дорогу", 1)[0].section == "0004"


def test_hybrid_merges_bonus_ranks_and_scores():
    bm25, vector, hybrid = retrievers()
    result = hybrid.search("перебои отопления", top_k=2)
    assert len({item.code for item in result}) == len(result)
    assert all(item.rank for item in result)
    assert all(0 <= item.hybridScore <= 1 for item in result)
    common = result[0]
    expected = 0.45 * (common.bm25Score or 0) + 0.55 * (
        common.vectorScore or 0
    )
    assert common.hybridScore >= min(1, expected + 0.05)


def test_section_filter_and_empty_query():
    bm25, vector, hybrid = retrievers()
    assert all(item.section == "0004" for item in bm25.search("ремонт", section="0004"))
    assert all(
        item.section == "0004"
        for item in vector.search("ремонт", section="0004")
    )
    assert hybrid.search("!!!") == []
    assert hybrid.search("ремонт", section="9999") == []


def test_bm25_and_faiss_round_trip():
    output = Path("data/evaluation/test-output")
    paths = [
        output / "bm25.pkl",
        output / "faiss.index",
        output / "embeddings.npy",
        output / "metadata.json",
    ]
    bm25, vector, _ = retrievers()
    try:
        bm25.save(paths[0])
        assert BM25ThemeRetriever.load(paths[0], THEMES).search("ремонт", 1)
        vector.save(paths[1], paths[2], paths[3])
        loaded = FaissThemeRetriever.load(
            THEMES, paths[1], paths[2], paths[3],
            "text-embedding-3-small", embedding_fn=embedding,
        )
        assert loaded.search("отопление", 1)
    finally:
        for path in paths:
            if path.exists():
                path.unlink()


def test_bm25_rejects_different_theme_order():
    path = Path("data/evaluation/test-output/bm25.pkl")
    bm25, _, _ = retrievers()
    try:
        bm25.save(path)
        with pytest.raises(ValueError, match="does not match"):
            BM25ThemeRetriever.load(path, list(reversed(THEMES)))
    finally:
        if path.exists():
            path.unlink()
