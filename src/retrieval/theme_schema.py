from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ThemeRecord(BaseModel):
    code: str
    name: str
    section: str
    sectionName: str | None = None
    parentCode: str | None = None
    parentName: str | None = None
    level: int | None = None
    isLeaf: bool = True
    path: list[str] = Field(default_factory=list)
    pathCodes: list[str] = Field(default_factory=list)
    classifierVersion: str = "2025-10-31"


class ThemeCandidate(BaseModel):
    code: str
    name: str
    section: str
    sectionName: str | None = None
    parentCode: str | None = None
    parentName: str | None = None
    path: list[str] = Field(default_factory=list)
    pathCodes: list[str] = Field(default_factory=list)
    classifierVersion: str = "2025-10-31"
    bm25Score: float | None = None
    vectorScore: float | None = None
    hybridScore: float
    rank: int
    source: Literal["bm25", "vector", "hybrid"]


def candidate_from_theme(theme: ThemeRecord, **scores: object) -> ThemeCandidate:
    fields = {
        "code": theme.code,
        "name": theme.name,
        "section": theme.section,
        "sectionName": theme.sectionName,
        "parentCode": theme.parentCode,
        "parentName": theme.parentName,
        "path": theme.path,
        "pathCodes": theme.pathCodes,
        "classifierVersion": theme.classifierVersion,
    }
    fields.update(scores)
    return ThemeCandidate(**fields)
