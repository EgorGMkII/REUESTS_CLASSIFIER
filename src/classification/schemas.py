from pydantic import BaseModel, Field

CLASSIFIER_VERSION = "2025-10-31"


class ThemePrediction(BaseModel):
    code: str
    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    section: str


class QuestionTypePrediction(BaseModel):
    code: str
    name: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(default="", exclude=True)


class QuestionSubtypePrediction(BaseModel):
    code: str
    officialCode: str
    name: str
    confidence: float = Field(ge=0.0, le=1.0)


class ClassificationMeta(BaseModel):
    requestId: str
    processedAt: str
    sourceTextLength: int
    extractedFromFiles: bool = False


class ClassificationResult(BaseModel):
    themes: list[ThemePrediction]
    questionType: QuestionTypePrediction
    questionSubtype: QuestionSubtypePrediction
    classifierVersion: str = CLASSIFIER_VERSION
    meta: ClassificationMeta
