from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from src.api.errors import ApiException
from src.classification.pipeline import ClassificationPipeline
from tools.ocr_lab.hydra_vision_ocr import (
    call_vision_api,
    chunks,
    compact_text,
    preprocess_images,
    render_pdf_pages,
    sum_usage,
)

logger = logging.getLogger("requests_classifier.api")
MAX_PDF_UPLOAD_BYTES = 25 * 1024 * 1024
DEFAULT_HYDRA_BASE_URL = "https://api.hydraai.ru/v1"
DEFAULT_HYDRA_MODEL = "gpt-5-mini"


class PdfClassificationService:
    def __init__(
        self,
        pipeline: ClassificationPipeline | None,
        work_dir: Path = Path("data/web_uploads"),
        dpi: int = 120,
        preprocess: str = "grayscale-contrast",
        pages_per_request: int = 5,
        max_tokens: int = 2500,
        timeout: int = 300,
    ):
        self.pipeline = pipeline
        self.work_dir = work_dir
        self.dpi = dpi
        self.preprocess = preprocess
        self.pages_per_request = pages_per_request
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_key = os.getenv("HYDRA_API_KEY") or ""
        self.base_url = (os.getenv("HYDRA_BASE_URL") or DEFAULT_HYDRA_BASE_URL).rstrip("/")
        self.model = os.getenv("HYDRA_MODEL") or DEFAULT_HYDRA_MODEL

    def classify_pdf(self, pdf_path: Path, request_id: str) -> dict[str, Any]:
        if self.pipeline is None:
            raise ApiException(503, "CLASSIFIER_NOT_READY", "Classifier is not ready")
        if pdf_path.suffix.lower() != ".pdf":
            raise ApiException(422, "VALIDATION_ERROR", "Only PDF files are supported")

        run_dir = self.work_dir / request_id
        ocr_dir = run_dir / "ocr"
        cleanup_dir = run_dir / "llm-cleaned"
        ocr_dir.mkdir(parents=True, exist_ok=True)
        cleanup_dir.mkdir(parents=True, exist_ok=True)

        try:
            cleaned_text, ocr_debug = self._run_ocr(pdf_path, ocr_dir)
            (cleanup_dir / "cleaned_text.txt").write_text(
                cleaned_text, encoding="utf-8"
            )
            (cleanup_dir / "cleanup_debug.json").write_text(
                json.dumps(
                    {
                        "fallbackUsed": False,
                        "reason": "hydra_vision_ocr_direct",
                        "inputLength": len(cleaned_text),
                        "outputLength": len(cleaned_text),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            if not cleaned_text.strip():
                raise ApiException(
                    422,
                    "OCR_TEXT_EMPTY",
                    "Could not extract meaningful text from PDF",
                )

            result = self.pipeline.run(cleaned_text)
            result = result.model_copy(
                update={
                    "meta": result.meta.model_copy(
                        update={
                            "requestId": request_id,
                            "sourceTextLength": len(cleaned_text),
                            "extractedFromFiles": True,
                        }
                    )
                }
            )
            return {
                "result": result,
                "candidates": self._candidates(),
                "text": {
                    "cleanedPreview": cleaned_text[:2000],
                    "ocrLength": len(cleaned_text),
                    "cleanedLength": len(cleaned_text),
                    "cleanupFallbackUsed": False,
                    "ocrUsage": ocr_debug.get("usage"),
                },
                "artifacts": {
                    "runDir": str(run_dir),
                    "ocrText": str(ocr_dir / "text.txt"),
                    "cleanedText": str(cleanup_dir / "cleaned_text.txt"),
                },
            }
        except ApiException:
            raise
        except Exception as error:
            logger.exception(
                "pdf_classification_failed",
                extra={"requestId": request_id, "pdfPath": str(pdf_path)},
            )
            raise ApiException(
                500,
                "PDF_CLASSIFICATION_FAILED",
                "Could not classify PDF",
                {"errorType": type(error).__name__},
            ) from error

    def save_upload(self, content: bytes, request_id: str, filename: str) -> Path:
        if not content:
            raise ApiException(422, "VALIDATION_ERROR", "Uploaded PDF is empty")
        if len(content) > MAX_PDF_UPLOAD_BYTES:
            raise ApiException(
                413,
                "PDF_TOO_LARGE",
                "Uploaded PDF is too large",
                {"maxBytes": MAX_PDF_UPLOAD_BYTES},
            )
        if not content.lstrip().startswith(b"%PDF"):
            raise ApiException(422, "VALIDATION_ERROR", "Uploaded file is not a PDF")
        suffix = Path(filename or "upload.pdf").suffix or ".pdf"
        if suffix.lower() != ".pdf":
            raise ApiException(422, "VALIDATION_ERROR", "Only PDF files are supported")
        run_dir = self.work_dir / request_id
        run_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = run_dir / f"source{suffix.lower()}"
        pdf_path.write_bytes(content)
        return pdf_path

    def _run_ocr(self, pdf_path: Path, out_dir: Path) -> tuple[str, dict[str, Any]]:
        if not self.api_key:
            raise ApiException(
                503,
                "OCR_API_KEY_NOT_SET",
                "HYDRA_API_KEY is not set",
            )

        pages_dir = out_dir / "pages"
        processed_pages_dir = out_dir / "pages_preprocessed"
        image_paths = render_pdf_pages(pdf_path, pages_dir, self.dpi)
        ocr_image_paths = preprocess_images(
            image_paths, processed_pages_dir, self.preprocess
        )

        text_parts: list[str] = []
        batches: list[dict[str, Any]] = []
        for batch_index, batch in enumerate(
            chunks(ocr_image_paths, self.pages_per_request)
        ):
            start_page = batch_index * self.pages_per_request + 1
            text, debug = call_vision_api(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                image_paths=batch,
                start_page=start_page,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
            text_parts.append(text)
            batches.append(debug)

        text = compact_text(text_parts)
        raw = {
            "sourcePdf": str(pdf_path),
            "dpi": self.dpi,
            "preprocess": self.preprocess,
            "pageCount": len(image_paths),
            "pagesPerRequest": self.pages_per_request,
            "baseUrl": self.base_url,
            "model": self.model,
            "outputLength": len(text),
            "usage": sum_usage(batches),
            "batches": batches,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "hydra_ocr_debug.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "text.txt").write_text(text, encoding="utf-8")
        return text, raw

    def _candidates(self) -> dict[str, Any]:
        theme_selector_diagnostics = (
            getattr(self.pipeline, "last_theme_selector_diagnostics", None) or {}
        )
        theme_candidates = []
        for candidate in getattr(self.pipeline, "last_candidates", [])[:10]:
            theme_candidates.append(
                {
                    "code": candidate.code,
                    "name": candidate.name,
                    "section": candidate.section,
                    "retrievalScore": candidate.hybridScore,
                    "rank": candidate.rank,
                }
            )

        pair_candidates = []
        for candidate in getattr(self.pipeline, "last_question_pair_candidates", []):
            score = min(
                candidate.question_type.confidence,
                candidate.question_subtype.confidence,
            )
            pair_candidates.append(
                {
                    "key": candidate.key,
                    "confidence": score,
                    "questionType": candidate.question_type,
                    "questionSubtype": candidate.question_subtype,
                }
            )
        pair_candidates.sort(key=lambda item: item["confidence"], reverse=True)
        return {
            "themes": theme_candidates,
            "themeSelection": theme_selector_diagnostics,
            "questionPairs": pair_candidates[:3],
        }
