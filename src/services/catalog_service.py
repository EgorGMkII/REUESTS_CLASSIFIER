from __future__ import annotations

from typing import Any

from src.classification.question_type_classifier import QUESTION_TYPES
from src.repositories.catalog_repository import CatalogRepository

from src.api.schemas import (
    CatalogResponse,
    QuestionSubtypeCatalogResponse,
    QuestionSubtypeCatalogItem,
    QuestionTypeCatalogItem,
    ThemeCatalogResponse,
)


class CatalogService:
    def __init__(self, repository: CatalogRepository):
        self.repository = repository
        self._themes = repository.load_theme_leafs()
        self._theme_tree = repository.load_theme_tree()
        self._subtypes = repository.load_question_subtypes()

    @property
    def classifier_version(self) -> str:
        return self._themes[0].classifierVersion if self._themes else ""

    @property
    def subtype_catalog_version(self) -> str:
        return self._subtypes.catalogVersion

    @property
    def themes_count(self) -> int:
        return len(self._themes)

    def themes(self) -> ThemeCatalogResponse:
        return ThemeCatalogResponse(
            classifierVersion=self.classifier_version,
            leafCount=len(self._themes),
            tree=self._theme_tree,
        )

    def question_types(self) -> list[QuestionTypeCatalogItem]:
        return [
            QuestionTypeCatalogItem(code=code, name=name)
            for code, name in sorted(QUESTION_TYPES.items(), key=lambda item: int(item[0]))
        ]

    def question_subtypes(self) -> QuestionSubtypeCatalogResponse:
        return QuestionSubtypeCatalogResponse(
            catalogVersion=self._subtypes.catalogVersion,
            items=[
                QuestionSubtypeCatalogItem(
                    code=item.code,
                    officialCode=item.officialCode,
                    name=item.name,
                    allowedQuestionTypes=item.allowedQuestionTypes,
                )
                for item in self._subtypes.items
            ],
        )

    def catalog(self) -> CatalogResponse:
        question_types = self.question_types()
        question_subtypes = self.question_subtypes()
        themes = self.themes()
        return CatalogResponse(
            classifierVersion=self.classifier_version,
            subtypeCatalogVersion=self.subtype_catalog_version,
            themesCount=len(self._themes),
            questionTypesCount=len(question_types),
            questionSubtypesCount=len(question_subtypes.items),
            themes=themes,
            questionTypes=question_types,
            questionSubtypes=question_subtypes,
        )

    def readiness_metadata(self) -> dict[str, Any]:
        return {
            "classifierVersion": self.classifier_version,
            "subtypeCatalogVersion": self.subtype_catalog_version,
            "themesCount": len(self._themes),
        }
