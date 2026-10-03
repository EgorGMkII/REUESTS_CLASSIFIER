from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data" / "evaluation" / "real_v2.json"
DEFAULT_OUTPUT = ROOT / "data" / "evaluation" / "real_v3.json"
DEFAULT_PDF_DIR = ROOT / "Обращения"
DEFAULT_RUNS_DIR = ROOT / "data" / "evaluation" / "hydra_ocr_runs"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build real_v3.json by extracting appeal text through Hydra vision OCR"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--preprocess", choices=["none", "grayscale-contrast"], default="grayscale-contrast")
    parser.add_argument("--pages-per-request", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run Hydra OCR even when text.txt already exists",
    )
    return parser


def document_number(case_id: str) -> int:
    match = re.fullmatch(r"real-(\d+)", case_id)
    if not match:
        raise ValueError(f"Unexpected case id: {case_id}")
    return int(match.group(1))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def display_path(path: Path) -> str:
    resolved = resolve_path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def run_command(command: list[str]) -> None:
    print("RUN:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def run_hydra_ocr(
    pdf_path: Path,
    out_dir: Path,
    dpi: int,
    preprocess: str,
    pages_per_request: int,
    max_tokens: int,
    timeout: int,
    force: bool,
) -> Path:
    text_path = out_dir / "text.txt"
    if text_path.exists() and not force:
        return text_path

    run_command(
        [
            sys.executable,
            "tools/ocr_lab/hydra_vision_ocr.py",
            "--pdf",
            str(pdf_path),
            "--out-dir",
            str(out_dir),
            "--dpi",
            str(dpi),
            "--preprocess",
            preprocess,
            "--pages-per-request",
            str(pages_per_request),
            "--max-tokens",
            str(max_tokens),
            "--timeout",
            str(timeout),
        ]
    )
    return text_path


def update_tags(case: dict[str, Any]) -> list[str]:
    tags = list(case.get("tags", []))
    for tag in ["real", "hydra-ocr", "real-v3"]:
        if tag not in tags:
            tags.append(tag)
    return [tag for tag in tags if tag not in {"ocr-cleaned", "real-v2"}]


def main() -> int:
    args = _parser().parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    pdf_dir = resolve_path(args.pdf_dir)
    runs_dir = resolve_path(args.runs_dir)

    dataset = load_json(input_path)
    cases = dataset.get("cases", [])
    if not isinstance(cases, list):
        raise SystemExit("Input dataset must contain a cases list")

    selected_cases = cases[: args.limit] if args.limit is not None else cases
    updated_cases: list[dict[str, Any]] = []

    for index, case in enumerate(selected_cases, start=1):
        case_id = case.get("id")
        if not isinstance(case_id, str):
            raise SystemExit(f"Case #{index} has invalid id")

        number = document_number(case_id)
        pdf_path = pdf_dir / f"{number}.pdf"
        if not pdf_path.exists():
            raise SystemExit(f"PDF for {case_id} not found: {pdf_path}")

        out_dir = runs_dir / case_id
        print(f"[{index}/{len(selected_cases)}] {case_id}: {pdf_path}", flush=True)
        text_path = run_hydra_ocr(
            pdf_path=pdf_path,
            out_dir=out_dir,
            dpi=args.dpi,
            preprocess=args.preprocess,
            pages_per_request=args.pages_per_request,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            force=args.force,
        )
        text = text_path.read_text(encoding="utf-8").strip()

        updated_case = dict(case)
        updated_case["text"] = text
        updated_case["tags"] = update_tags(updated_case)
        updated_case["source"] = {
            "pdf": display_path(pdf_path),
            "hydraOcrText": display_path(text_path),
            "hydraOcrDebug": display_path(out_dir / "hydra_ocr_debug.json"),
        }
        updated_cases.append(updated_case)

    if args.limit is not None and len(cases) > len(selected_cases):
        updated_cases.extend(cases[len(selected_cases) :])

    output = dict(dataset)
    output["datasetVersion"] = "3.0"
    output["cases"] = updated_cases
    save_json(output_path, output)
    print(f"Saved dataset: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
