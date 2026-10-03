from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any

import requests


DEFAULT_BASE_URL = "https://api.hydraai.ru/v1"
DEFAULT_MODEL = "gpt-5-mini"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract appeal text from a PDF via Hydra/OpenAI-compatible vision API"
    )
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--preprocess", choices=["none", "grayscale-contrast"], default="none")
    parser.add_argument("--pages-per-request", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
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
            image_path = pages_dir / f"page_{index:03d}.jpg"
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
            processed = ImageEnhance.Contrast(processed).enhance(1.5)
            processed = ImageEnhance.Sharpness(processed).enhance(1.2)
            processed = processed.filter(ImageFilter.SHARPEN)
        else:
            processed = image

        processed_path = out_dir / image_path.name
        processed.save(processed_path, quality=88)
        processed_paths.append(processed_path)

    return processed_paths


def image_to_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def chunks(items: list[Path], size: int) -> list[list[Path]]:
    size = max(1, size)
    return [items[index : index + size] for index in range(0, len(items), size)]


def api_config(args: argparse.Namespace) -> tuple[str, str, str]:
    api_key = os.getenv("HYDRA_API_KEY")
    if not api_key:
        raise SystemExit("HYDRA_API_KEY is not set.")

    base_url = (
        args.base_url
        or os.getenv("HYDRA_BASE_URL")
        or DEFAULT_BASE_URL
    ).rstrip("/")
    model = args.model or os.getenv("HYDRA_MODEL") or DEFAULT_MODEL
    return api_key, base_url, model


def prompt_for_pages(start_page: int, end_page: int) -> str:
    return f"""Распознай документ на страницах {start_page}-{end_page}.

Нужно вернуть не весь OCR подряд, а только исходное обращение заявителя/гражданина
в виде плотного текста, пригодного для дальнейшей классификации и тематического поиска.

Главное правило:
- если в PDF есть обращение гражданина и сопроводительное письмо/ответ органа,
  верни только обращение гражданина;
- ответ органа, пересылку, резолюцию и служебную маршрутизацию не пересказывай.

Не включай:
- сопроводительные письма органов: "направляю для рассмотрения", "просим уведомить заявителя";
- ответы прокуратуры, администрации, министерств и иных ведомств;
- регистрационные карточки, шапки приема, QR/штрихкоды, подписи сканера, номера страниц;
- перечни адресатов, приложения, служебные реквизиты, повторяющиеся дубли;
- длинные цитаты законов и политические преамбулы, если они не нужны для понимания сути.

Что сохранить:
- кто обращается, если это важно: житель, родители, работники, коллективное обращение;
- предмет проблемы, объект, адрес/место, организации, даты, номера документов, суммы;
- действия/бездействие, последствия, что именно просит заявитель;
- предметные ключевые слова для поиска темы: полигон ТБО, свалка, забой скота,
  невыплата зарплаты, дорога, автобус, отопление, жилье, субсидия, школа и т.п.
- форму документа, если она влияет на вид обращения: обычное обращение гражданина,
  запрос прокуратуры, запрос органа, служебный/межведомственный запрос.

Если документ выглядит как запрос прокуратуры, запрос органа или служебное обращение:
- если в шапке/реквизитах видно, что это именно запрос прокуратуры или прокурора,
  первой строкой строго напиши: "Запрос прокуратуры.";
- если это запрос другого органа или служебный/межведомственный запрос,
  первой строкой напиши: "Форма документа: запрос органа/служебный запрос";
- сохрани формулировки вроде "прокуратура просит", "прокуратура запрашивает",
  "просим предоставить сведения", если они есть;
- не превращай такой документ в обычную жалобу гражданина.
- не пиши "Обращение в прокуратуру", если документ исходит от прокуратуры;
  "обращение в прокуратуру" допустимо только для обращения гражданина/организации,
  адресованного прокуратуре.

Не сглаживай предмет обращения до общих формулировок вроде:
"нарушение прав", "просьба о содействии", "незаконные действия".

Если обращение длинное:
- не переписывай весь текст полностью;
- верни плотную смысловую выжимку до 2500–4000 символов;
- сохрани максимум полезных предметных слов для тематического retrieval.

Если фрагмент неразборчив, напиши [неразборчиво].
Не классифицируй обращение и не добавляй выводов от себя.
Если на страницах нет текста обращения, верни пустую строку.

Формат ответа:
Краткое тело обращения:
...

Ключевые слова для тематического поиска:
..."""


def call_vision_api(
    api_key: str,
    base_url: str,
    model: str,
    image_paths: list[Path],
    start_page: int,
    max_tokens: int,
    timeout: int,
) -> tuple[str, dict[str, Any]]:
    end_page = start_page + len(image_paths) - 1
    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt_for_pages(start_page, end_page)}
    ]
    content.extend(
        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}}
        for image_path in image_paths
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )

    debug: dict[str, Any] = {
        "pages": [str(path) for path in image_paths],
        "startPage": start_page,
        "endPage": end_page,
        "statusCode": response.status_code,
    }

    try:
        response_payload = response.json()
    except ValueError:
        response_payload = {"rawText": response.text[:2000]}

    debug["response"] = response_payload
    debug["usage"] = extract_usage(response_payload)

    if response.status_code >= 400:
        message = response_payload if isinstance(response_payload, dict) else response.text
        raise RuntimeError(f"Hydra OCR request failed: {response.status_code} {message}")

    text = extract_response_text(response_payload)
    return text, debug


def extract_usage(payload: Any) -> dict[str, int | None]:
    if not isinstance(payload, dict) or not isinstance(payload.get("usage"), dict):
        return {
            "promptTokens": None,
            "completionTokens": None,
            "totalTokens": None,
        }

    usage = payload["usage"]
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    return {
        "promptTokens": prompt_tokens if isinstance(prompt_tokens, int) else None,
        "completionTokens": completion_tokens if isinstance(completion_tokens, int) else None,
        "totalTokens": total_tokens if isinstance(total_tokens, int) else None,
    }


def sum_usage(batches: list[dict[str, Any]]) -> dict[str, int | None]:
    totals: dict[str, int | None] = {
        "promptTokens": 0,
        "completionTokens": 0,
        "totalTokens": 0,
    }
    has_any_usage = False

    for batch in batches:
        usage = batch.get("usage")
        if not isinstance(usage, dict):
            continue

        for key in totals:
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] = int(totals[key] or 0) + value
                has_any_usage = True

    if not has_any_usage:
        return {
            "promptTokens": None,
            "completionTokens": None,
            "totalTokens": None,
        }
    return totals


def extract_response_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise ValueError("API response is not a JSON object")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("API response has no choices")

    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise ValueError("API response choice has no message")

    content = message.get("content")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "\n".join(parts).strip()

    raise ValueError("API response message content is not text")


def compact_text(parts: list[str]) -> str:
    lines: list[str] = []
    previous_blank = False

    for part in parts:
        for raw_line in part.splitlines():
            line = raw_line.strip()
            if not line:
                if not previous_blank and lines:
                    lines.append("")
                previous_blank = True
                continue
            lines.append(line)
            previous_blank = False

    return "\n".join(lines).strip()


def main() -> int:
    args = _parser().parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")

    api_key, base_url, model = api_config(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pages_dir = args.out_dir / "pages"
    processed_pages_dir = args.out_dir / "pages_preprocessed"
    image_paths = render_pdf_pages(args.pdf, pages_dir, args.dpi)
    ocr_image_paths = preprocess_images(image_paths, processed_pages_dir, args.preprocess)

    text_parts: list[str] = []
    debug_batches: list[dict[str, Any]] = []

    for batch_index, batch in enumerate(chunks(ocr_image_paths, args.pages_per_request)):
        start_page = batch_index * args.pages_per_request + 1
        text, debug = call_vision_api(
            api_key=api_key,
            base_url=base_url,
            model=model,
            image_paths=batch,
            start_page=start_page,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
        text_parts.append(text)
        debug_batches.append(debug)

    final_text = compact_text(text_parts)
    total_usage = sum_usage(debug_batches)
    debug = {
        "sourcePdf": str(args.pdf),
        "dpi": args.dpi,
        "preprocess": args.preprocess,
        "pageCount": len(image_paths),
        "pagesPerRequest": args.pages_per_request,
        "baseUrl": base_url,
        "model": model,
        "outputLength": len(final_text),
        "usage": total_usage,
        "batches": debug_batches,
    }

    (args.out_dir / "text.txt").write_text(final_text, encoding="utf-8")
    (args.out_dir / "hydra_ocr_debug.json").write_text(
        json.dumps(debug, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved Hydra OCR text: {args.out_dir / 'text.txt'}")
    print(f"Saved Hydra OCR debug: {args.out_dir / 'hydra_ocr_debug.json'}")
    if total_usage["totalTokens"] is not None:
        print(
            "Token usage: "
            f"input={total_usage['promptTokens']}, "
            f"output={total_usage['completionTokens']}, "
            f"total={total_usage['totalTokens']}"
        )
    else:
        print("Token usage: not returned by API")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
