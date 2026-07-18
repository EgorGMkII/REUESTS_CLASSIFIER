from __future__ import annotations

from pydantic import BaseModel, Field


class ThemeNode(BaseModel):
    code: str
    name: str
    section: str
    sectionName: str | None = None
    parentCode: str | None = None
    parentName: str | None = None
    level: int
    isLeaf: bool = False
    path: list[str] = Field(default_factory=list)
    pathCodes: list[str] = Field(default_factory=list)
    classifierVersion: str = "2025-10-31"


class ImportReport(BaseModel):
    classifierVersion: str = "2025-10-31"
    inputFile: str
    totalParsedRows: int = 0
    totalValidNodes: int = 0
    totalLeafNodes: int = 0
    sections: dict[str, int] = Field(
        default_factory=lambda: {f"{number:04d}": 0 for number in range(1, 6)}
    )
    duplicates: list[dict] = Field(default_factory=list)
    invalidCodes: list[dict] = Field(default_factory=list)
    emptyNames: list[dict] = Field(default_factory=list)
    fiveBlockCodes: list[dict] = Field(default_factory=list)
    suspiciousFootnoteCodes: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
