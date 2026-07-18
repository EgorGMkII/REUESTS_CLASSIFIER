from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.classification.schemas import (
    ClassificationResult,
    QuestionSubtypePrediction,
    QuestionTypePrediction,
)

MAX_TEXT_LENGTH = 20_000


class ClassificationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)


class ReadinessResponse(BaseModel):
    ready: bool
    classifierVersion: str
    subtypeCatalogVersion: str
    themesCount: int
    embeddingModel: str


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    requestId: str
    error: ErrorBody


class QuestionTypeCatalogItem(BaseModel):
    code: str
    name: str


class QuestionSubtypeCatalogItem(BaseModel):
    code: str
    officialCode: str
    name: str
    allowedQuestionTypes: list[str]


class QuestionSubtypeCatalogResponse(BaseModel):
    catalogVersion: str
    items: list[QuestionSubtypeCatalogItem]


class ThemeCatalogResponse(BaseModel):
    classifierVersion: str
    leafCount: int
    tree: list[dict[str, Any]]


class CatalogResponse(BaseModel):
    classifierVersion: str
    subtypeCatalogVersion: str
    themesCount: int
    questionTypesCount: int
    questionSubtypesCount: int
    themes: ThemeCatalogResponse
    questionTypes: list[QuestionTypeCatalogItem]
    questionSubtypes: QuestionSubtypeCatalogResponse


class ThemeCandidateResponse(BaseModel):
    code: str
    name: str
    section: str
    retrievalScore: float
    rank: int


class QuestionPairCandidateResponse(BaseModel):
    key: str
    confidence: float
    questionType: QuestionTypePrediction
    questionSubtype: QuestionSubtypePrediction


class ClassificationCandidateResponse(BaseModel):
    themes: list[ThemeCandidateResponse]
    questionPairs: list[QuestionPairCandidateResponse]


class PdfTextResponse(BaseModel):
    cleanedPreview: str
    ocrLength: int
    cleanedLength: int
    cleanupFallbackUsed: bool
    ocrUsage: dict[str, Any] | None = None


class PdfArtifactsResponse(BaseModel):
    runDir: str
    ocrText: str
    cleanedText: str


class PdfClassificationResponse(BaseModel):
    result: ClassificationResult
    candidates: ClassificationCandidateResponse
    text: PdfTextResponse
    artifacts: PdfArtifactsResponse
