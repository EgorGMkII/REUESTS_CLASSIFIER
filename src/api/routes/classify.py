from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from src.api.errors import get_request_id
from src.api.schemas import ClassificationRequest, PdfClassificationResponse
from src.classification.schemas import ClassificationResult
from src.services.classification_service import ClassificationService
from src.services.pdf_classification_service import PdfClassificationService

router = APIRouter(prefix="/api/v1", tags=["classification"])


def get_classification_service(request: Request) -> ClassificationService:
    return request.app.state.classification_service


def get_pdf_classification_service(request: Request) -> PdfClassificationService:
    return request.app.state.pdf_classification_service


@router.post("/classify", response_model=ClassificationResult)
def classify(
    body: ClassificationRequest,
    request: Request,
    service: ClassificationService = Depends(get_classification_service),
) -> ClassificationResult:
    return service.classify(body.text, get_request_id(request))


@router.post("/classify/pdf", response_model=PdfClassificationResponse)
async def classify_pdf(
    request: Request,
    service: PdfClassificationService = Depends(get_pdf_classification_service),
) -> dict:
    request_id = get_request_id(request)
    filename = request.headers.get("X-Filename", "upload.pdf")
    content = await request.body()
    pdf_path = service.save_upload(content, request_id, filename)
    return service.classify_pdf(pdf_path, request_id)
