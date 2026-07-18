from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .theme_schema import ImportReport, ThemeNode
from .theme_validation import (
    detect_suspicious_footnote_code,
    get_level,
    get_parent_code,
    is_five_block_code,
    is_four_block_code,
    is_working_classifier_code,
    normalize_code,
    normalize_name,
)

CLASSIFIER_VERSION = "2025-10-31"
CODE_TOKEN_RE = re.compile(
    r"(?<!\d)(?:\d{4}\.){3}\d{4,5}(?:\.\d{4})?(?!\d)"
)
METHODICAL_HEADING_RE = re.compile(
    r"\s+0000\.[^. \s]+\.[0-9]{4}\.[0-9]{4}\s*[–—-]\s*"
    r"ПЕРЕЧЕНЬ\s+ТЕМАТИК.*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class RawThemeRow:
    code: str
    name: str


def remove_methodical_heading(name: str) -> str:
    """Remove a methodical-list heading accidentally joined to a node name."""
    return normalize_name(METHODICAL_HEADING_RE.sub("", name))


def _docx_text(path: Path) -> str:
    try:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as error:
        raise RuntimeError(
            "Reading .docx requires python-docx: pip install python-docx"
        ) from error

    document = Document(path)
    lines: list[str] = []
    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            lines.append(Paragraph(child, document).text)
        elif child.tag.endswith("}tbl"):
            table = Table(child, document)
            for row in table.rows:
                lines.extend(cell.text for cell in row.cells)
    return "\n".join(lines)


def extract_text_from_word(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input Word document does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return _docx_text(path)
    if suffix != ".doc":
        raise ValueError("Only .doc and .docx input files are supported")

    executable = shutil.which("libreoffice") or shutil.which("soffice")
    if not executable:
        raise RuntimeError(
            "A legacy .doc file requires LibreOffice conversion. Install "
            "LibreOffice and ensure 'libreoffice' or 'soffice' is on PATH, "
            "or save the document as .docx."
        )
    with tempfile.TemporaryDirectory(prefix="themes-import-") as temp_dir:
        result = subprocess.run(
            [
                executable,
                "--headless",
                "--convert-to",
                "docx",
                "--outdir",
                temp_dir,
                str(path.resolve()),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        converted = Path(temp_dir) / f"{path.stem}.docx"
        if result.returncode or not converted.exists():
            details = normalize_name(result.stderr or result.stdout)
            raise RuntimeError(f"LibreOffice could not convert {path}: {details}")
        return _docx_text(converted)


def extract_code_name_pairs(text: str) -> list[RawThemeRow]:
    matches = list(CODE_TOKEN_RE.finditer(text.replace("\xa0", " ")))
    rows: list[RawThemeRow] = []
    for index, match in enumerate(matches):
        name_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        rows.append(
            RawThemeRow(
                code=normalize_code(match.group()),
                name=normalize_name(text[match.end() : name_end]),
            )
        )
    return rows


def _copy_node(node: ThemeNode, **updates: Any) -> ThemeNode:
    if hasattr(node, "model_copy"):
        return node.model_copy(update=updates)
    return node.copy(update=updates)


def build_theme_nodes(rows: list[RawThemeRow]) -> list[ThemeNode]:
    nodes: list[ThemeNode] = []
    seen: set[str] = set()
    for row in rows:
        code = normalize_code(row.code)
        name = remove_methodical_heading(row.name)
        if not is_working_classifier_code(code) or not name or code in seen:
            continue
        seen.add(code)
        nodes.append(
            ThemeNode(
                code=code,
                name=name,
                section=code.split(".")[0],
                parentCode=get_parent_code(code),
                level=get_level(code),
            )
        )
    return sorted(nodes, key=lambda node: tuple(int(part) for part in node.code.split(".")))


def build_paths(nodes: list[ThemeNode]) -> list[ThemeNode]:
    by_code = {node.code: node for node in nodes}
    result: list[ThemeNode] = []
    for node in nodes:
        chain: list[ThemeNode] = []
        current: ThemeNode | None = node
        visited: set[str] = set()
        while current is not None and current.code not in visited:
            visited.add(current.code)
            chain.append(current)
            current = by_code.get(current.parentCode) if current.parentCode else None
        chain.reverse()
        section_code = f"{node.section}.0000.0000.0000"
        section_node = by_code.get(section_code)
        parent = by_code.get(node.parentCode) if node.parentCode else None
        result.append(
            _copy_node(
                node,
                sectionName=section_node.name if section_node else None,
                parentName=parent.name if parent else None,
                path=[item.name for item in chain],
                pathCodes=[item.code for item in chain],
            )
        )
    return result


def mark_leaf_nodes(nodes: list[ThemeNode], leaf_mode: str) -> list[ThemeNode]:
    if leaf_mode not in {"four-block-only", "strict"}:
        raise ValueError(f"Unknown leaf mode: {leaf_mode}")
    all_parents = {node.parentCode for node in nodes if node.parentCode}
    four_block_parents = {
        node.parentCode
        for node in nodes
        if node.parentCode and is_four_block_code(node.code)
    }
    parents = all_parents if leaf_mode == "strict" else four_block_parents
    return [_copy_node(node, isLeaf=node.code not in parents) for node in nodes]


def build_tree(nodes: list[ThemeNode]) -> list[dict]:
    items = {
        node.code: {"code": node.code, "name": node.name, "children": []}
        for node in nodes
    }
    roots: list[dict] = []
    for node in nodes:
        item = items[node.code]
        if node.parentCode in items:
            items[node.parentCode]["children"].append(item)
        else:
            roots.append(item)
    return roots


def _model_dump(model: Any) -> Any:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return model


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [_model_dump(item) for item in data] if isinstance(data, list) else (
        _model_dump(data) if hasattr(data, "dict") or hasattr(data, "model_dump") else data
    )
    path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def import_themes(
    input_path: Path,
    out_dir: Path,
    leaf_mode: str = "four-block-only",
    fix_footnotes: bool = False,
) -> tuple[list[ThemeNode], list[ThemeNode], ImportReport]:
    text = extract_text_from_word(input_path)
    raw_rows = extract_code_name_pairs(text)
    report = ImportReport(inputFile=str(input_path), totalParsedRows=len(raw_rows))
    accepted: list[RawThemeRow] = []
    seen: dict[str, str] = {}

    for row in raw_rows:
        code = normalize_code(row.code)
        suspicious = detect_suspicious_footnote_code(code)
        if suspicious:
            suspicious["fixed"] = fix_footnotes
            report.suspiciousFootnoteCodes.append(suspicious)
            if fix_footnotes:
                code = suspicious["suggestedCode"]
        if is_five_block_code(code):
            report.fiveBlockCodes.append({"code": code, "name": row.name})
        if not (is_four_block_code(code) or is_five_block_code(code)):
            report.invalidCodes.append({"code": code, "name": row.name})
            continue
        if not is_working_classifier_code(code):
            continue
        if not row.name:
            report.emptyNames.append({"code": code})
            continue
        if code in seen:
            report.duplicates.append(
                {"code": code, "firstName": seen[code], "duplicateName": row.name}
            )
            continue
        seen[code] = row.name
        accepted.append(RawThemeRow(code, row.name))

    nodes = mark_leaf_nodes(build_paths(build_theme_nodes(accepted)), leaf_mode)
    by_code = {node.code for node in nodes}
    for node in nodes:
        if node.parentCode and node.parentCode not in by_code:
            report.warnings.append(
                f"Missing parent {node.parentCode} for {node.code}"
            )
    leaf_nodes = [
        node for node in nodes if node.isLeaf and is_four_block_code(node.code)
    ]
    report.totalValidNodes = len(nodes)
    report.totalLeafNodes = len(leaf_nodes)
    report.sections = {
        section: sum(node.section == section for node in nodes)
        for section in report.sections
    }

    outputs = {
        "themes_all.json": nodes,
        "themes_leaf.json": leaf_nodes,
        "themes_tree.json": build_tree(nodes),
        "themes_import_report.json": report,
    }
    for filename, data in outputs.items():
        write_json(out_dir / filename, data)
    return nodes, leaf_nodes, report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import thematic classifier from Word")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/Классификаторы на 31.10.2025.doc"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/classifiers"))
    parser.add_argument(
        "--leaf-mode",
        choices=("four-block-only", "strict"),
        default="four-block-only",
    )
    parser.add_argument(
        "--fix-footnotes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    nodes, leaves, report = import_themes(
        args.input, args.out_dir, args.leaf_mode, args.fix_footnotes
    )
    print("Import completed.")
    print(f"Total valid nodes: {len(nodes)}")
    print(f"Total leaf nodes: {len(leaves)}")
    print("Sections:")
    for section, count in report.sections.items():
        print(f"  {section}: {count}")
    print(f"Warnings: {len(report.warnings)}")
    print("Output:")
    for filename in (
        "themes_all.json",
        "themes_leaf.json",
        "themes_tree.json",
        "themes_import_report.json",
    ):
        print(f"  {args.out_dir / filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
