from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classification._llm import LLMCallable, invoke_default_llm, parse_json_response
from src.classification.text_preprocessor import compact_ocr_text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean OCR text with an LLM before classification"
    )
    parser.add_argument("--input", type=Path, required=True, help="OCR text.txt path")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-input-chars", type=int, default=12_000)
    parser.add_argument("--max-output-chars", type=int, default=6_000)
    return parser


class LLMOcrTextCleaner:
    def __init__(
        self,
        llm: LLMCallable | None = None,
        max_input_chars: int = 12_000,
        max_output_chars: int = 6_000,
    ):
        self._llm = llm or invoke_default_llm
        self._max_input_chars = max_input_chars
        self._max_output_chars = max_output_chars

    def clean(self, text: str) -> tuple[str, dict[str, Any]]:
        compacted = compact_ocr_text(text, self._max_input_chars)
        if not compacted:
            return "", {"fallbackUsed": False, "reason": "empty_input"}

        prompt = self._prompt(compacted)

        try:
            payload = parse_json_response(self._llm(prompt))
            cleaned = payload.get("cleanedText")
            notes = payload.get("notes", [])

            if not isinstance(cleaned, str):
                raise ValueError("cleanedText must be a string")

            cleaned = compact_ocr_text(cleaned, self._max_output_chars)
            if len(cleaned) < 10:
                raise ValueError("cleanedText is too short")

            return cleaned, {
                "fallbackUsed": False,
                "inputLength": len(text),
                "compactedInputLength": len(compacted),
                "outputLength": len(cleaned),
                "notes": notes if isinstance(notes, list) else [],
            }
        except Exception as exc:
            fallback = compact_ocr_text(compacted, self._max_output_chars)
            return fallback, {
                "fallbackUsed": True,
                "reason": type(exc).__name__,
                "message": str(exc),
                "inputLength": len(text),
                "outputLength": len(fallback),
            }

    def _prompt(self, text: str) -> str:
        return f"""Ты очищаешь OCR-текст обращения гражданина перед классификацией.

Задача: выделить тело самого обращения и исправить только очевидные OCR-артефакты.

Что нужно сделать:
- убрать шапки, служебные отметки, номера страниц, подписи, контакты, случайные OCR-символы;
- сохранить факты, просьбы, жалобы, объекты, места, учреждения, диагнозы и важные обстоятельства;
- исправить очевидные OCR-ошибки, если восстановление почти однозначно;
- не добавлять новые факты;
- если слово неясно, оставить его как есть или аккуратно пометить в тексте как неясное;
- не классифицировать обращение и не добавлять коды;
- писать связным русским текстом, пригодным для передачи в retrieval/classification pipeline.

Примеры допустимых исправлений:
- "субсидия начбем" → "субсидия на съем";
- "приобретения жилья" → "приобретение жилья", если по смыслу это просьба/вопрос;
- "онимевшая" → "онемевшая".

Верни только JSON без markdown:
{{
  "cleanedText": "очищенный текст обращения",
  "notes": ["короткие замечания о сомнительных исправлениях, если есть"]
}}

OCR-текст:
{text}"""


def main() -> int:
    args = _parser().parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    source_text = args.input.read_text(encoding="utf-8")
    cleaner = LLMOcrTextCleaner(
        max_input_chars=args.max_input_chars,
        max_output_chars=args.max_output_chars,
    )
    cleaned_text, debug = cleaner.clean(source_text)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cleaned_text.txt").write_text(cleaned_text, encoding="utf-8")
    (args.out_dir / "cleanup_debug.json").write_text(
        json.dumps(
            {
                "sourcePath": str(args.input),
                **debug,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved cleaned text: {args.out_dir / 'cleaned_text.txt'}")
    print(f"Saved cleanup debug: {args.out_dir / 'cleanup_debug.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
