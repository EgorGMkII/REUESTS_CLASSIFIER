from __future__ import annotations

import re
import unicodedata

FOUR_BLOCK_RE = re.compile(r"^\d{4}(?:\.\d{4}){3}$")
FIVE_BLOCK_RE = re.compile(r"^\d{4}(?:\.\d{4}){4}$")
SUSPICIOUS_FOOTNOTE_RE = re.compile(r"^\d{4}(?:\.\d{4}){2}\.\d{5}$")
WORKING_SECTIONS = {f"{number:04d}" for number in range(1, 6)}


def normalize_code(code: str) -> str:
    value = unicodedata.normalize("NFKC", code)
    value = value.replace("\xa0", "").replace(" ", "")
    return value.strip(".,;:")


def normalize_name(name: str) -> str:
    value = unicodedata.normalize("NFC", name).replace("\xa0", " ")
    return re.sub(r"\s+", " ", value).strip()


def is_four_block_code(code: str) -> bool:
    return bool(FOUR_BLOCK_RE.fullmatch(normalize_code(code)))


def is_five_block_code(code: str) -> bool:
    return bool(FIVE_BLOCK_RE.fullmatch(normalize_code(code)))


def is_working_classifier_code(code: str) -> bool:
    code = normalize_code(code)
    return (
        is_four_block_code(code) or is_five_block_code(code)
    ) and code.split(".", 1)[0] in WORKING_SECTIONS


def detect_suspicious_footnote_code(code: str) -> dict | None:
    raw = normalize_code(code)
    if not SUSPICIOUS_FOOTNOTE_RE.fullmatch(raw):
        return None
    blocks = raw.split(".")
    blocks[-1] = blocks[-1][:-1]
    return {
        "rawCode": raw,
        "suggestedCode": ".".join(blocks),
        "reason": "last block has 5 digits, likely footnote marker",
    }


def get_level(code: str) -> int:
    code = normalize_code(code)
    if not (is_four_block_code(code) or is_five_block_code(code)):
        raise ValueError(f"Invalid classifier code: {code}")
    return sum(block != "0000" for block in code.split("."))


def get_parent_code(code: str) -> str | None:
    code = normalize_code(code)
    if is_five_block_code(code):
        return ".".join(code.split(".")[:-1])
    if not is_four_block_code(code):
        raise ValueError(f"Invalid classifier code: {code}")
    parts = code.split(".")
    nonzero = [index for index, part in enumerate(parts) if part != "0000"]
    if len(nonzero) <= 1:
        return None
    parts[nonzero[-1]] = "0000"
    return ".".join(parts)
