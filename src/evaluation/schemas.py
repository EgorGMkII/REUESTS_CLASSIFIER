from __future__ import annotations

from pydantic import BaseModel, Field

from src.classification.schemas import ClassificationResult


class ExpectedTheme(BaseModel):
    code: str
    name: str
    section: str


class ExpectedLabel(BaseModel):
    code: str
    name: str


class ExpectedSubtype(ExpectedLabel):
    officialCode: str


class ExpectedClassification(BaseModel):
    themes: list[ExpectedTheme] = Field(min_length=1)
    questionType: ExpectedLabel
    questionSubtype: ExpectedSubtype


class EvaluationCase(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    expected: ExpectedClassification
    tags: list[str] = Field(default_factory=list)


class EvaluationDataset(BaseModel):
    datasetVersion: str
    classifierVersion: str
    cases: list[EvaluationCase] = Field(min_length=1)


class RetrievalDiagnostics(BaseModel):
    expectedThemeRanks: dict[str, int | None] = Field(default_factory=dict)
    topCandidateCodes: list[str] = Field(default_factory=list)


class RetrievalQueryDiagnostics(BaseModel):
    section: str | None = None
    query: str
    topCandidateCodes: list[str] = Field(default_factory=list)


class TextDiagnostics(BaseModel):
    sourceLength: int = Field(ge=0)
    preprocessedLength: int = Field(ge=0)
    normalizedLength: int = Field(ge=0)
    preprocessedText: str = ""
    normalizedText: str = ""
    typeDecisionText: str = ""
    retrievalQueries: list[RetrievalQueryDiagnostics] = Field(default_factory=list)


class QuestionTypeDiagnostics(BaseModel):
    code: str
    name: str
    confidence: float
    reason: str = ""


class QuestionPairDiagnostics(BaseModel):
    key: str
    questionTypeCode: str
    questionTypeName: str
    questionSubtypeCode: str
    questionSubtypeOfficialCode: str
    questionSubtypeName: str


class PredictionRecord(BaseModel):
    id: str
    prediction: ClassificationResult | None = None
    latencyMs: float = Field(ge=0)
    fallbackUsed: bool = False
    error: str | None = None
    retrieval: RetrievalDiagnostics | None = None
    textDiagnostics: TextDiagnostics | None = None
    questionTypeCandidates: list[QuestionTypeDiagnostics] = Field(default_factory=list)
    questionPairCandidates: list[QuestionPairDiagnostics] = Field(default_factory=list)
