from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from src.classification.subtype_catalog import QuestionSubtypeCatalog, load_subtype_catalog
from src.retrieval.hybrid_theme_retriever import load_themes
from src.retrieval.theme_schema import ThemeRecord


class CatalogRepository(Protocol):
    def load_theme_leafs(self) -> list[ThemeRecord]: ...

    def load_theme_tree(self) -> list[dict[str, Any]]: ...

    def load_question_subtypes(self) -> QuestionSubtypeCatalog: ...


class JsonCatalogRepository:
    def __init__(self, themes_path: Path, themes_tree_path: Path, subtypes_path: Path):
        self.themes_path = Path(themes_path)
        self.themes_tree_path = Path(themes_tree_path)
        self.subtypes_path = Path(subtypes_path)

    def load_theme_leafs(self) -> list[ThemeRecord]:
        return load_themes(self.themes_path)

    def load_theme_tree(self) -> list[dict[str, Any]]:
        return json.loads(self.themes_tree_path.read_text(encoding="utf-8"))

    def load_question_subtypes(self) -> QuestionSubtypeCatalog:
        return load_subtype_catalog(self.subtypes_path)
