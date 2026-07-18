from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from src.classification.pipeline import ClassificationPipeline, build_pipeline
from src.classification.subtype_catalog import DEFAULT_SUBTYPE_CATALOG_PATH
from src.repositories.catalog_repository import JsonCatalogRepository
from src.services.catalog_service import CatalogService
from src.services.classification_service import ClassificationService
from src.services.pdf_classification_service import PdfClassificationService

from .errors import register_exception_handlers
from .middleware import RequestContextMiddleware
from .routes import catalog, classify, health
from .schemas import ReadinessResponse

DEFAULT_THEMES_PATH = Path("data/classifiers/themes_leaf.json")
DEFAULT_THEMES_TREE_PATH = Path("data/classifiers/themes_tree.json")
DEFAULT_INDEX_DIR = Path("data/indexes/themes")
STATIC_DIR = Path("src/api/static")


def _origins() -> list[str]:
    raw = os.getenv(
        "CLASSIFIER_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def create_app(
    pipeline: ClassificationPipeline | None = None,
    themes_path: Path = DEFAULT_THEMES_PATH,
    index_dir: Path = DEFAULT_INDEX_DIR,
    subtypes_path: Path = DEFAULT_SUBTYPE_CATALOG_PATH,
    themes_tree_path: Path = DEFAULT_THEMES_TREE_PATH,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        initialization_error = None
        catalog_error = None
        catalog_service = None
        try:
            catalog_service = CatalogService(
                JsonCatalogRepository(themes_path, themes_tree_path, subtypes_path)
            )
            application.state.catalog_service = catalog_service
        except Exception as error:
            application.state.catalog_service = None
            catalog_error = f"{type(error).__name__}: {error}"

        metadata = {}
        try:
            metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
        except Exception as error:
            initialization_error = f"{type(error).__name__}: {error}"

        if pipeline is not None:
            application.state.pipeline = pipeline
        elif initialization_error is None:
            try:
                application.state.pipeline = build_pipeline(
                    themes_path, index_dir, subtypes_path
                )
            except Exception as error:
                application.state.pipeline = None
                initialization_error = f"{type(error).__name__}: {error}"
        else:
            application.state.pipeline = None

        application.state.classification_service = ClassificationService(
            application.state.pipeline
        )
        application.state.pdf_classification_service = PdfClassificationService(
            application.state.pipeline
        )
        if catalog_service is not None:
            catalog_metadata = catalog_service.readiness_metadata()
            application.state.readiness = ReadinessResponse(
                ready=initialization_error is None,
                classifierVersion=catalog_metadata["classifierVersion"],
                subtypeCatalogVersion=catalog_metadata["subtypeCatalogVersion"],
                themesCount=catalog_metadata["themesCount"],
                embeddingModel=metadata.get("embeddingModel", ""),
            )
        else:
            application.state.readiness = ReadinessResponse(
                ready=False,
                classifierVersion="",
                subtypeCatalogVersion="",
                themesCount=0,
                embeddingModel=metadata.get("embeddingModel", ""),
            )
        application.state.initialization_error = initialization_error or catalog_error
        yield

    application = FastAPI(
        title="Citizen Request Classifier",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.add_middleware(RequestContextMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )
    register_exception_handlers(application)
    if STATIC_DIR.exists():
        application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @application.get("/", include_in_schema=False)
        def web_index():
            return FileResponse(STATIC_DIR / "index.html")

        @application.get("/favicon.ico", include_in_schema=False)
        def favicon():
            return Response(status_code=204)

    application.include_router(health.router)
    application.include_router(classify.router)
    application.include_router(catalog.router)
    return application


app = create_app()
