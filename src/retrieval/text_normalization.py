from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from .theme_schema import ThemeRecord

_NON_WORD = re.compile(r"[^a-zа-я0-9]+", re.IGNORECASE)
_TOKEN = re.compile(r"[a-zа-я0-9]+", re.IGNORECASE)


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "").lower().replace("ё", "е")
    value = value.replace("\xa0", " ")
    # Hyphens and slashes are semantic word boundaries for lexical retrieval.
    value = re.sub(r"[-‐‑‒–—/\\]+", " ", value)
    return re.sub(r"\s+", " ", _NON_WORD.sub(" ", value)).strip()


@lru_cache(maxsize=1)
def _russian_tools():
    try:
        from pymorphy3 import MorphAnalyzer
        from razdel import tokenize

        return tokenize, MorphAnalyzer()
    except (ImportError, RuntimeError):
        return None


@lru_cache(maxsize=50_000)
def _lemmatize(token: str, morph: object) -> str:
    return morph.parse(token)[0].normal_form


def tokenize_ru(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    tools = _russian_tools()
    if tools is None:
        return _TOKEN.findall(normalized)
    tokenizer, morph = tools
    return [
        _lemmatize(token.text, morph)
        for token in tokenizer(normalized)
        if _TOKEN.fullmatch(token.text)
    ]


def make_theme_search_document(theme: ThemeRecord) -> str:
    parts = [
        theme.name,
        theme.name,
        theme.name,
        theme.parentName or "",
        theme.parentName or "",
        theme.sectionName or "",
        " ".join(theme.path),
    ]
    return normalize_text(" ".join(part for part in parts if part))
