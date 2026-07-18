from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, model_validator

DEFAULT_SUBTYPE_CATALOG_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "classifiers"
    / "question_subtypes.json"
)
QUESTION_TYPE_CODES = {str(number) for number in range(1, 10)}


class QuestionSubtypeDefinition(BaseModel):
    code: str
    officialCode: str
    name: str
    allowedQuestionTypes: list[str] = Field(min_length=1)


class QuestionSubtypeCatalog(BaseModel):
    catalogVersion: str
    items: list[QuestionSubtypeDefinition]

    @model_validator(mode="after")
    def validate_catalog(self) -> "QuestionSubtypeCatalog":
        codes = [item.code for item in self.items]
        if len(codes) != len(set(codes)):
            raise ValueError("Subtype catalog contains duplicate codes")
        if len(self.items) != 39:
            raise ValueError(f"Subtype catalog must contain 39 items, got {len(self.items)}")
        for item in self.items:
            unknown = set(item.allowedQuestionTypes) - QUESTION_TYPE_CODES
            if unknown:
                raise ValueError(
                    f"Subtype {item.code} has unknown question types: {sorted(unknown)}"
                )
        dash = next((item for item in self.items if item.code == "-"), None)
        if dash is None or set(dash.allowedQuestionTypes) != QUESTION_TYPE_CODES:
            raise ValueError("Subtype '-' must be available for all question types")
        covered = {
            question_type
            for item in self.items
            for question_type in item.allowedQuestionTypes
        }
        if covered != QUESTION_TYPE_CODES:
            raise ValueError("Subtype catalog does not cover all question types")
        return self

    def by_code(self) -> dict[str, QuestionSubtypeDefinition]:
        return {item.code: item for item in self.items}

    def allowed_codes(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for question_type in sorted(QUESTION_TYPE_CODES):
            if question_type in {"5", "6", "7", "8", "9"}:
                result[question_type] = ["-"]
                continue
            result[question_type] = [
                item.code
                for item in self.items
                if item.code != "-" and question_type in item.allowedQuestionTypes
            ]
        return result


class SubtypeCatalogRepository(Protocol):
    def load(self) -> QuestionSubtypeCatalog: ...


class JsonSubtypeCatalogRepository:
    def __init__(self, path: Path = DEFAULT_SUBTYPE_CATALOG_PATH):
        self.path = Path(path)

    def load(self) -> QuestionSubtypeCatalog:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return QuestionSubtypeCatalog.model_validate(data)


def load_subtype_catalog(
    path: Path = DEFAULT_SUBTYPE_CATALOG_PATH,
) -> QuestionSubtypeCatalog:
    return JsonSubtypeCatalogRepository(path).load()
