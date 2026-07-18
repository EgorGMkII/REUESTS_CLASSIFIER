from __future__ import annotations

import logging

from openai import APIError

from src.classification.errors import ClassificationValidationError
from src.classification.pipeline import ClassificationPipeline
from src.classification.schemas import ClassificationResult

from src.api.errors import ApiException

logger = logging.getLogger("requests_classifier.api")


class ClassificationService:
    def __init__(self, pipeline: ClassificationPipeline | None):
        self.pipeline = pipeline

    def classify(self, text: str, request_id: str) -> ClassificationResult:
        if self.pipeline is None:
            raise ApiException(503, "CLASSIFIER_NOT_READY", "Classifier is not ready")
        if not text.strip():
            raise ApiException(422, "VALIDATION_ERROR", "Text must not be blank")
        try:
            result = self.pipeline.run(text)
        except ClassificationValidationError as error:
            raise ApiException(
                422,
                "CLASSIFICATION_VALIDATION_ERROR",
                "Classification result validation failed",
                {"errors": error.errors},
            ) from error
        except (ConnectionError, TimeoutError, APIError) as error:
            raise ApiException(
                503,
                "PROVIDER_UNAVAILABLE",
                "Classification provider is unavailable",
            ) from error
        meta = result.meta.model_copy(update={"requestId": request_id})
        updated = result.model_copy(update={"meta": meta})
        logger.info(
            "classification_completed",
            extra={
                "requestId": request_id,
                "textLength": len(text),
                "themes": [theme.code for theme in updated.themes],
                "questionType": updated.questionType.code,
                "questionSubtype": updated.questionSubtype.code,
                "fallbackUsed": self._fallback_used(updated),
            },
        )
        return updated

    @staticmethod
    def _fallback_used(result: ClassificationResult) -> bool:
        return bool(
            any(theme.confidence == 0.4 for theme in result.themes)
            or result.questionType.confidence == 0.3
            or result.questionSubtype.confidence == 0.3
        )
