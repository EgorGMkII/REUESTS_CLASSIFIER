from __future__ import annotations

from fastapi import APIRouter, Request

from src.api.errors import ApiException
from src.api.schemas import ReadinessResponse

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", response_model=ReadinessResponse)
def ready(request: Request) -> ReadinessResponse:
    readiness = getattr(request.app.state, "readiness", None)
    if readiness is None or not readiness.ready:
        raise ApiException(
            503,
            "CLASSIFIER_NOT_READY",
            getattr(request.app.state, "initialization_error", "Classifier is not ready"),
        )
    return readiness
