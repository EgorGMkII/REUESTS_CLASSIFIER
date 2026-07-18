from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from src.api.errors import ApiException
from src.api.schemas import (
    CatalogResponse,
    QuestionSubtypeCatalogResponse,
    QuestionTypeCatalogItem,
    ThemeCatalogResponse,
)
from src.services.catalog_service import CatalogService

router = APIRouter(prefix="/api/v1/catalog", tags=["catalog"])


def get_catalog_service(request: Request) -> CatalogService:
    service = getattr(request.app.state, "catalog_service", None)
    if service is None:
        raise ApiException(503, "CATALOG_NOT_READY", "Catalog is not ready")
    return service


@router.get("", response_model=CatalogResponse)
def catalog(service: CatalogService = Depends(get_catalog_service)) -> CatalogResponse:
    return service.catalog()


@router.get("/themes", response_model=ThemeCatalogResponse)
def themes(service: CatalogService = Depends(get_catalog_service)) -> ThemeCatalogResponse:
    return service.themes()


@router.get("/question-types", response_model=list[QuestionTypeCatalogItem])
def question_types(
    service: CatalogService = Depends(get_catalog_service),
) -> list[QuestionTypeCatalogItem]:
    return service.question_types()


@router.get("/question-subtypes", response_model=QuestionSubtypeCatalogResponse)
def question_subtypes(
    service: CatalogService = Depends(get_catalog_service),
) -> QuestionSubtypeCatalogResponse:
    return service.question_subtypes()
