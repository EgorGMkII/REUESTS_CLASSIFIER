from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "evaluation" / "real_v1.json"
DEFAULT_OUTPUT = ROOT / "data" / "evaluation" / "real_v2.json"
DEFAULT_PDF_DIR = ROOT / "Обращения"
DEFAULT_RUNS_DIR = ROOT / "data" / "evaluation" / "ocr_runs"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build real_v2.json by OCR + LLM-cleaning matching real_v1 cases"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--lang", default="ru")
    parser.add_argument(
        "--textline-orientation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable PaddleOCR text line orientation detection",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def document_number(case_id: str) -> int:
    match = re.fullmatch(r"real-(\d+)", case_id)
    if not match:
        raise ValueError(f"Unexpected case id: {case_id}")
    return int(match.group(1))


def run_command(command: list[str]) -> None:
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def run_ocr(
    pdf_path: Path,
    out_dir: Path,
    lang: str,
    dpi: int,
    textline_orientation: bool,
) -> Path:
    text_path = out_dir / "text.txt"
    if text_path.exists():
        return text_path

    command = [
        sys.executable,
        "tools/ocr_lab/paddleocr_pdf.py",
        "--pdf",
        str(pdf_path),
        "--out-dir",
        str(out_dir),
        "--lang",
        lang,
        "--dpi",
        str(dpi),
        "--preprocess",
        "grayscale-contrast",
    ]
    if textline_orientation:
        command.append("--textline-orientation")
    run_command(command)
    return text_path


def run_cleanup(ocr_text_path: Path, out_dir: Path) -> Path:
    cleaned_path = out_dir / "cleaned_text.txt"
    if cleaned_path.exists():
        return cleaned_path

    run_command(
        [
            sys.executable,
            "tools/ocr_lab/ocr_llm_cleanup.py",
            "--input",
            str(ocr_text_path),
            "--out-dir",
            str(out_dir),
        ]
    )
    return cleaned_path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = _parser().parse_args()
    dataset = load_json(args.input)
    cases = dataset.get("cases", [])
    if not isinstance(cases, list):
        raise SystemExit("Input dataset must contain a cases list")

    updated_cases: list[dict[str, Any]] = []
    selected_cases = cases[: args.limit] if args.limit is not None else cases

    for index, case in enumerate(selected_cases, start=1):
        case_id = case.get("id")
        if not isinstance(case_id, str):
            raise SystemExit(f"Case #{index} has invalid id")

        number = document_number(case_id)
        pdf_path = args.pdf_dir / f"{number}.pdf"
        if not pdf_path.exists():
            raise SystemExit(f"PDF for {case_id} not found: {pdf_path}")

        case_run_dir = args.runs_dir / case_id
        ocr_dir = case_run_dir / "ocr"
        cleanup_dir = case_run_dir / "llm-cleaned"

        if args.force and case_run_dir.exists():
            # Keep this script non-destructive by default; force only recomputes files
            # that downstream commands overwrite themselves.
            pass

        print(f"[{index}/{len(selected_cases)}] {case_id}: {pdf_path}", flush=True)
        ocr_text_path = run_ocr(
            pdf_path,
            ocr_dir,
            args.lang,
            args.dpi,
            args.textline_orientation,
        )
        cleaned_text_path = run_cleanup(ocr_text_path, cleanup_dir)
        cleaned_text = cleaned_text_path.read_text(encoding="utf-8").strip()

        updated_case = dict(case)
        updated_case["text"] = cleaned_text
        tags = list(updated_case.get("tags", []))
        for tag in ["real", "ocr", "ocr-cleaned", "real-v2"]:
            if tag not in tags:
                tags.append(tag)
        updated_case["tags"] = tags
        updated_case["source"] = {
            "pdf": str(pdf_path.relative_to(ROOT)),
            "ocrText": str(ocr_text_path.relative_to(ROOT)),
            "cleanedText": str(cleaned_text_path.relative_to(ROOT)),
        }
        updated_cases.append(updated_case)

    if args.limit is not None and len(cases) > len(selected_cases):
        updated_cases.extend(cases[len(selected_cases) :])

    output = dict(dataset)
    output["datasetVersion"] = "2.0"
    output["cases"] = updated_cases
    save_json(args.output, output)
    print(f"Saved dataset: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
