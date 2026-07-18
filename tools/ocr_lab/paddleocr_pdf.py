from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PaddleOCR on a PDF file")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lang", default="ru")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--textline-orientation",
        action="store_true",
        help="Enable PaddleOCR text line orientation detection",
    )
    parser.add_argument(
        "--preprocess",
        choices=["none", "grayscale-contrast"],
        default="none",
        help="Optional image preprocessing before OCR",
    )
    return parser


def render_pdf_pages(pdf_path: Path, pages_dir: Path, dpi: int) -> list[Path]:
    try:
        import fitz
    except ImportError as exc:
        raise SystemExit(
            "PyMuPDF is not installed. Run: python -m pip install pymupdf"
        ) from exc

    pages_dir.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    image_paths: list[Path] = []
    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc, start=1):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = pages_dir / f"page_{index:03d}.png"
            pixmap.save(image_path)
            image_paths.append(image_path)
    return image_paths


def preprocess_images(image_paths: list[Path], out_dir: Path, mode: str) -> list[Path]:
    if mode == "none":
        return image_paths

    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    except ImportError as exc:
        raise SystemExit("Pillow is not installed. Run: python -m pip install pillow") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    processed_paths: list[Path] = []

    for image_path in image_paths:
        image = Image.open(image_path)

        if mode == "grayscale-contrast":
            processed = ImageOps.grayscale(image)
            processed = ImageOps.autocontrast(processed)
            processed = ImageEnhance.Contrast(processed).enhance(1.6)
            processed = ImageEnhance.Sharpness(processed).enhance(1.3)
            processed = processed.filter(ImageFilter.SHARPEN)
        else:
            processed = image

        processed_path = out_dir / image_path.name
        processed.save(processed_path)
        processed_paths.append(processed_path)

    return processed_paths


def build_ocr(lang: str, textline_orientation: bool) -> Any:
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise SystemExit(
            "PaddleOCR is not installed. Run: python -m pip install paddleocr"
        ) from exc

    init_variants = [
        {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": textline_orientation,
            "enable_mkldnn": False,
            "lang": lang,
        },
        {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": textline_orientation,
            "lang": lang,
        },
        {
            "use_textline_orientation": textline_orientation,
            "enable_mkldnn": False,
            "lang": lang,
        },
        {"use_textline_orientation": textline_orientation, "lang": lang},
        {"use_angle_cls": True, "enable_mkldnn": False, "lang": lang},
        {"use_angle_cls": True, "lang": lang},
        {"lang": lang},
    ]

    last_error: Exception | None = None
    for kwargs in init_variants:
        try:
            return PaddleOCR(**kwargs)
        except (TypeError, ValueError) as exc:
            last_error = exc

    raise RuntimeError(f"Could not initialize PaddleOCR: {last_error}") from last_error


def run_ocr(ocr: Any, image_path: Path) -> Any:
    if hasattr(ocr, "predict"):
        return ocr.predict(str(image_path))

    try:
        return ocr.ocr(str(image_path), cls=True)
    except TypeError:
        return ocr.ocr(str(image_path))


def _is_ocr_line(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 2
        and isinstance(value[1], (list, tuple))
        and len(value[1]) >= 2
        and isinstance(value[1][0], str)
    )


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def extract_lines(result: Any) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []

    if result is None:
        return lines

    candidates = result
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        candidates = result[0]

    if isinstance(result, list) and result and isinstance(result[0], dict):
        for page_result in result:
            lines.extend(extract_lines(page_result))
        return lines

    if isinstance(result, dict):
        rec_texts = result.get("rec_texts") or result.get("texts") or []
        rec_scores = result.get("rec_scores") or result.get("scores") or []
        rec_boxes = (
            result.get("rec_polys")
            or result.get("dt_polys")
            or result.get("rec_boxes")
            or result.get("boxes")
            or []
        )
        for index, text in enumerate(rec_texts):
            score = rec_scores[index] if index < len(rec_scores) else None
            box = rec_boxes[index] if index < len(rec_boxes) else None
            lines.append(
                {
                    "text": str(text),
                    "score": to_jsonable(score),
                    "box": to_jsonable(box),
                }
            )
        return lines

    if not isinstance(candidates, list):
        return lines

    for item in candidates:
        if _is_ocr_line(item):
            box = item[0]
            text = item[1][0]
            score = item[1][1]
            lines.append(
                {
                    "text": text,
                    "score": float(score),
                    "box": to_jsonable(box),
                }
            )
        elif isinstance(item, list):
            for nested in item:
                if _is_ocr_line(nested):
                    box = nested[0]
                    text = nested[1][0]
                    score = nested[1][1]
                    lines.append(
                        {
                            "text": text,
                            "score": float(score),
                            "box": to_jsonable(box),
                        }
                    )

    return lines


def main() -> int:
    args = _parser().parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = args.out_dir / "pages"
    processed_pages_dir = args.out_dir / "pages_preprocessed"

    image_paths = render_pdf_pages(args.pdf, pages_dir, args.dpi)
    ocr_image_paths = preprocess_images(image_paths, processed_pages_dir, args.preprocess)
    ocr = build_ocr(args.lang, args.textline_orientation)

    pages: list[dict[str, Any]] = []
    text_parts: list[str] = []

    for page_number, image_path in enumerate(ocr_image_paths, start=1):
        result = run_ocr(ocr, image_path)
        lines = extract_lines(result)
        page_text = "\n".join(line["text"] for line in lines)
        text_parts.append(f"--- PAGE {page_number} ---\n{page_text}".strip())
        pages.append(
            {
                "page": page_number,
                "imagePath": str(image_paths[page_number - 1]),
                "ocrImagePath": str(image_path),
                "lineCount": len(lines),
                "lines": lines,
            }
        )

    raw = {
        "sourcePdf": str(args.pdf),
        "dpi": args.dpi,
        "lang": args.lang,
        "preprocess": args.preprocess,
        "textlineOrientation": args.textline_orientation,
        "pageCount": len(pages),
        "pages": pages,
    }

    (args.out_dir / "raw_lines.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "text.txt").write_text(
        "\n\n".join(text_parts),
        encoding="utf-8",
    )

    print(f"Saved OCR text: {args.out_dir / 'text.txt'}")
    print(f"Saved raw OCR lines: {args.out_dir / 'raw_lines.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
