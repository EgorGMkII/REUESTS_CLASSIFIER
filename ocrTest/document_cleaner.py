import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CleanConfig:
    remove_headers: bool = True
    remove_routing_blocks: bool = True
    normalize_spaces: bool = True
    keep_geography: bool = True
    use_soft_replacement: bool = True


class DocumentCleaner:

    def __init__(self, config: CleanConfig = CleanConfig()):
        self.config = config

        self.org_patterns = [
            r"министерств\w*",
            r"правительств\w*",
            r"прокуратур\w*",
            r"администрац\w*",
            r"департамент\w*",
            r"управлен\w*",
            r"комитет\w*",
            r"отдел\w*",
            r"федеральн\w*служб\w*",
            r"российской федерац\w*",
            r"президент\w*",
        ]

        self.routing_patterns = [
            r"направляется по принадлежности",
            r"в соответствии с",
            r"приложение:.*",
        ]

        self.contact_patterns = {
            r"\S+@\S+": "<EMAIL>",
            r"\+?\d[\d\s\-\(\)]{7,}\d": "<PHONE>",
            r"http\S+": "<URL>",
            r"www\.\S+": "<URL>",
        }

        self.noise_token_pattern = re.compile(
            r"[A-Za-zА-Яа-яЁё]*\d+[A-Za-z]+|\b[a-zA-Z0-9]{15,}\b"
        )

        self.geo_patterns = [
            r"\bмоскв\w*\b",
            r"\bновосибирск\w*\b",
            r"\bобласт\w*\b",
            r"\bкра\w*\b",
        ]


    def clean(self, text: str) -> Tuple[str, dict]:
        if not text:
            return "", {}

        raw = text
        lines = self._split_lines(text)

        stats = {
            "initial_lines": len(lines),
            "removed_lines": 0,
            "kept_lines": 0,
        }

        cleaned_lines = []

        for line in lines:
            original = line
            line = line.strip()

            if not line:
                continue

            # 1. remove routing / header lines
            if self.config.remove_headers and self._is_header(line):
                stats["removed_lines"] += 1
                continue

            if self.config.remove_routing_blocks and self._is_routing(line):
                stats["removed_lines"] += 1
                continue

            # 2. soft replacements (contacts)
            if self.config.use_soft_replacement:
                line = self._replace_contacts(line)

            # 3. noise tokens filtering
            line = self._remove_noise_tokens(line)

            # 4. optional geography handling
            if not self.config.keep_geography:
                line = self._remove_geo(line)

            if line.strip():
                cleaned_lines.append(line)
                stats["kept_lines"] += 1

        text = " ".join(cleaned_lines)

        if self.config.normalize_spaces:
            text = self._normalize_spaces(text)

        return text.strip(), stats


    def _is_header(self, line: str) -> bool:
        line_low = line.lower()

        score = 0
        for p in self.org_patterns:
            if re.search(p, line_low):
                score += 2

        # typical OCR headers = short + org keywords
        if len(line) < 40 and score > 0:
            return True

        return score >= 2

    def _is_routing(self, line: str) -> bool:
        line_low = line.lower()

        for p in self.routing_patterns:
            if re.search(p, line_low):
                return True

        # massive lists of authorities
        if line.count(",") > 3 and "прокурат" in line_low:
            return True

        return False

    def _replace_contacts(self, line: str) -> str:
        for pattern, token in self.contact_patterns.items():
            line = re.sub(pattern, token, line, flags=re.IGNORECASE)
        return line

    def _remove_noise_tokens(self, text: str) -> str:
        tokens = text.split()
        filtered = []

        for t in tokens:

            # mixed garbage tokens
            if self.noise_token_pattern.search(t):
                continue

            # OCR random strings
            if len(t) > 14 and len(set(t)) > 10:
                continue

            filtered.append(t)

        return " ".join(filtered)

    def _remove_geo(self, text: str) -> str:
        for p in self.geo_patterns:
            text = re.sub(p, " ", text, flags=re.IGNORECASE)
        return text


    def _split_lines(self, text: str) -> List[str]:
        return re.split(r"[\n\r]+", text)

    def _normalize_spaces(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()