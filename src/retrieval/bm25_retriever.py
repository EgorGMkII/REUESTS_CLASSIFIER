from __future__ import annotations

import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from .text_normalization import make_theme_search_document, tokenize_ru
from .theme_schema import ThemeCandidate, ThemeRecord, candidate_from_theme


class BM25ThemeRetriever:
    def __init__(self, themes: list[ThemeRecord]):
        self.themes = list(themes)
        self.tokenized_corpus = [
            tokenize_ru(make_theme_search_document(theme)) for theme in self.themes
        ]
        self.index = BM25Okapi(self.tokenized_corpus)

    def search(
        self, query: str, top_k: int = 50, section: str | None = None
    ) -> list[ThemeCandidate]:
        tokens = tokenize_ru(query)
        if not tokens or top_k <= 0:
            return []
        allowed = [
            index
            for index, theme in enumerate(self.themes)
            if section is None or theme.section == section
        ]
        if not allowed:
            return []
        raw_scores = self.index.get_scores(tokens)
        max_score = max((float(raw_scores[index]) for index in allowed), default=0.0)
        ranked = sorted(
            allowed,
            key=lambda index: (-float(raw_scores[index]), self.themes[index].code),
        )[:top_k]
        results = []
        for rank, index in enumerate(ranked, 1):
            normalized = float(raw_scores[index]) / max_score if max_score > 0 else 0.0
            results.append(
                candidate_from_theme(
                    self.themes[index],
                    bm25Score=max(0.0, min(1.0, normalized)),
                    hybridScore=max(0.0, min(1.0, normalized)),
                    rank=rank,
                    source="bm25",
                )
            )
        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "codes": [theme.code for theme in self.themes],
            "tokenized_corpus": self.tokenized_corpus,
            "index": self.index,
        }
        path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

    @classmethod
    def load(
        cls, path: Path, themes: list[ThemeRecord]
    ) -> "BM25ThemeRetriever":
        payload = pickle.loads(path.read_bytes())
        codes = [theme.code for theme in themes]
        if payload.get("codes") != codes:
            raise ValueError("BM25 index metadata does not match the supplied themes")
        instance = cls.__new__(cls)
        instance.themes = list(themes)
        instance.tokenized_corpus = payload["tokenized_corpus"]
        instance.index = payload["index"]
        return instance
