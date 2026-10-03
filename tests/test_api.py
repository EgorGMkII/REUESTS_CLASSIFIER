from fastapi.testclient import TestClient
from pathlib import Path

from src.api.app import create_app
from src.api.middleware import REQUEST_ID_HEADER
from src.classification.errors import ClassificationValidationError
from src.classification.schemas import (
    ClassificationMeta,
    ClassificationResult,
    QuestionSubtypePrediction,
    QuestionTypePrediction,
    ThemePrediction,
)


class FakePipeline:
    def run(self, text):
        return ClassificationResult(
            themes=[
                ThemePrediction(
                    code="theme", name="Theme", confidence=0.9, section="0001"
                )
            ],
            questionType=QuestionTypePrediction(
                code="2", name="Заявление", confidence=0.8
            ),
            questionSubtype=QuestionSubtypePrediction(
                code="-",
                officialCode="-",
                name="Не применяется",
                confidence=0.3,
            ),
            meta=ClassificationMeta(
                requestId="pipeline-generated",
                processedAt="2026-07-06T00:00:00Z",
                sourceTextLength=len(text),
            ),
        )


def test_api_classification_readiness_and_request_id():
    with TestClient(create_app(pipeline=FakePipeline())) as client:
        response = client.post(
            "/api/v1/classify",
            json={"text": "Обращение"},
            headers={REQUEST_ID_HEADER: "req-test"},
        )
        assert response.status_code == 200
        assert response.headers[REQUEST_ID_HEADER] == "req-test"
        payload = response.json()
        assert payload["meta"]["requestId"] == "req-test"
        assert payload["questionSubtype"]["officialCode"] == "-"
        assert client.get("/health").status_code == 200
        readiness = client.get("/ready")
        assert readiness.status_code == 200
        assert readiness.json()["themesCount"] == 1220
        assert readiness.json()["subtypeCatalogVersion"] == "2025-10-31"


def test_api_generates_request_id_when_absent():
    with TestClient(create_app(pipeline=FakePipeline())) as client:
        response = client.post("/api/v1/classify", json={"text": "Обращение"})
        assert response.status_code == 200
        generated = response.headers[REQUEST_ID_HEADER]
        assert len(generated) == 32
        assert response.json()["meta"]["requestId"] == generated


def test_api_rejects_blank_text_with_unified_error():
    with TestClient(create_app(pipeline=FakePipeline())) as client:
        response = client.post(
            "/api/v1/classify",
            json={"text": "   "},
            headers={REQUEST_ID_HEADER: "req-blank"},
        )
        assert response.status_code == 422
        assert response.headers[REQUEST_ID_HEADER] == "req-blank"
        assert response.json() == {
            "requestId": "req-blank",
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Text must not be blank",
                "details": {},
            },
        }


def test_api_maps_classification_validation_error():
    class InvalidPipeline:
        def run(self, text):
            raise ClassificationValidationError(["bad result"])

    with TestClient(create_app(pipeline=InvalidPipeline())) as client:
        response = client.post(
            "/api/v1/classify",
            json={"text": "text"},
            headers={REQUEST_ID_HEADER: "req-invalid"},
        )
        assert response.status_code == 422
        assert response.json()["requestId"] == "req-invalid"
        assert response.json()["error"]["code"] == "CLASSIFICATION_VALIDATION_ERROR"
        assert response.json()["error"]["details"] == {"errors": ["bad result"]}


def test_api_maps_provider_unavailable():
    class UnavailablePipeline:
        def run(self, text):
            raise TimeoutError("provider timeout")

    with TestClient(create_app(pipeline=UnavailablePipeline())) as client:
        response = client.post("/api/v1/classify", json={"text": "text"})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "PROVIDER_UNAVAILABLE"


def test_api_maps_internal_error_without_request_text():
    class BrokenPipeline:
        def run(self, text):
            raise RuntimeError("contains request secret")

    with TestClient(
        create_app(pipeline=BrokenPipeline()), raise_server_exceptions=False
    ) as client:
        response = client.post("/api/v1/classify", json={"text": "private text"})
        assert response.status_code == 500
        payload = response.json()
        assert payload["error"]["code"] == "INTERNAL_ERROR"
        assert "private text" not in response.text
        assert "contains request secret" not in response.text


def test_catalog_endpoints():
    with TestClient(create_app(pipeline=FakePipeline())) as client:
        catalog = client.get("/api/v1/catalog")
        assert catalog.status_code == 200
        payload = catalog.json()
        assert payload["themesCount"] == 1220
        assert payload["questionTypesCount"] == 9
        assert payload["questionSubtypesCount"] == 39

        themes = client.get("/api/v1/catalog/themes")
        assert themes.status_code == 200
        assert themes.json()["leafCount"] == 1220
        assert themes.json()["tree"]

        question_types = client.get("/api/v1/catalog/question-types")
        assert question_types.status_code == 200
        assert len(question_types.json()) == 9

        question_subtypes = client.get("/api/v1/catalog/question-subtypes")
        assert question_subtypes.status_code == 200
        subtypes = question_subtypes.json()
        assert subtypes["catalogVersion"] == "2025-10-31"
        assert len(subtypes["items"]) == 39
        assert any(item["code"] == "-" for item in subtypes["items"])


def test_web_index_is_served():
    with TestClient(create_app(pipeline=FakePipeline())) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Классификация обращения из PDF" in response.text


def test_pdf_classification_endpoint_uses_pdf_service():
    output_dir = Path("data/evaluation/test-output")
    output_dir.mkdir(parents=True, exist_ok=True)

    class FakePdfService:
        def save_upload(self, content, request_id, filename):
            path = output_dir / filename
            path.write_bytes(content)
            return path

        def classify_pdf(self, pdf_path, request_id):
            result = FakePipeline().run("cleaned text")
            result = result.model_copy(
                update={"meta": result.meta.model_copy(update={"requestId": request_id})}
            )
            return {
                "result": result,
                "candidates": {
                    "themeSelection": {
                        "selected": [{"code": "theme", "role": "core"}],
                        "rejected": [],
                    },
                    "themes": [
                        {
                            "code": "theme",
                            "name": "Theme",
                            "section": "0001",
                            "retrievalScore": 0.91,
                            "rank": 1,
                        }
                    ],
                    "questionPairs": [
                        {
                            "key": "2|-",
                            "confidence": 0.8,
                            "questionType": result.questionType,
                            "questionSubtype": result.questionSubtype,
                        }
                    ],
                },
                "text": {
                    "cleanedPreview": "cleaned text",
                    "ocrLength": 20,
                    "cleanedLength": 12,
                    "cleanupFallbackUsed": False,
                },
                "artifacts": {
                    "runDir": str(output_dir),
                    "ocrText": str(output_dir / "text.txt"),
                    "cleanedText": str(output_dir / "cleaned_text.txt"),
                },
            }

    with TestClient(create_app(pipeline=FakePipeline())) as client:
        client.app.state.pdf_classification_service = FakePdfService()
        response = client.post(
            "/api/v1/classify/pdf",
            content=b"%PDF-1.4",
            headers={
                REQUEST_ID_HEADER: "pdf-req",
                "Content-Type": "application/pdf",
                "X-Filename": "test.pdf",
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["result"]["meta"]["requestId"] == "pdf-req"
        assert payload["candidates"]["questionPairs"][0]["key"] == "2|-"
        assert payload["candidates"]["themes"][0]["retrievalScore"] == 0.91
        assert payload["candidates"]["themeSelection"]["selected"] == [
            {"code": "theme", "role": "core"}
        ]
        assert payload["text"]["cleanedPreview"] == "cleaned text"


def test_request_id_rejects_path_like_values():
    with TestClient(create_app(pipeline=FakePipeline())) as client:
        response = client.post(
            "/api/v1/classify",
            json={"text": "Обращение"},
            headers={REQUEST_ID_HEADER: "../bad/path"},
        )
        assert response.status_code == 200
        generated = response.headers[REQUEST_ID_HEADER]
        assert generated != "../bad/path"
        assert len(generated) == 32
        assert response.json()["meta"]["requestId"] == generated


def test_pdf_upload_rejects_non_pdf_before_service_classification():
    with TestClient(create_app(pipeline=FakePipeline())) as client:
        response = client.post(
            "/api/v1/classify/pdf",
            content=b"not a real pdf",
            headers={
                REQUEST_ID_HEADER: "pdf-invalid",
                "Content-Type": "application/pdf",
                "X-Filename": "test.pdf",
            },
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_pdf_upload_rejects_too_large_file():
    from src.services import pdf_classification_service

    content = b"%PDF" + b"x" * 8
    with TestClient(create_app(pipeline=FakePipeline())) as client:
        original = pdf_classification_service.MAX_PDF_UPLOAD_BYTES
        pdf_classification_service.MAX_PDF_UPLOAD_BYTES = 4
        try:
            response = client.post(
                "/api/v1/classify/pdf",
                content=content,
                headers={
                    REQUEST_ID_HEADER: "pdf-large",
                    "Content-Type": "application/pdf",
                    "X-Filename": "test.pdf",
                },
            )
        finally:
            pdf_classification_service.MAX_PDF_UPLOAD_BYTES = original
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "PDF_TOO_LARGE"


def test_production_app_reports_initialization_error_as_not_ready(monkeypatch):
    from src.api import app as api_app

    def fail_build_pipeline(*args, **kwargs):
        raise ValueError("Theme indexes are stale")

    monkeypatch.setattr(api_app, "build_pipeline", fail_build_pipeline)
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
        readiness = client.get("/ready")
        assert readiness.status_code == 503
        assert readiness.json()["error"]["code"] == "CLASSIFIER_NOT_READY"
        classify = client.post("/api/v1/classify", json={"text": "Обращение"})
        assert classify.status_code == 503
        assert classify.json()["error"]["code"] == "CLASSIFIER_NOT_READY"
        assert client.get("/api/v1/catalog").status_code == 200
