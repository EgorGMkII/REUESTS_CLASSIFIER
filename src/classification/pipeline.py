from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from src.retrieval.bm25_retriever import BM25ThemeRetriever
from src.retrieval.faiss_retriever import FaissThemeRetriever
from src.retrieval.hybrid_theme_retriever import HybridThemeRetriever, load_themes
from src.retrieval.theme_schema import ThemeCandidate
from src.retrieval.text_normalization import normalize_text

from .errors import ClassificationValidationError
from .question_pair_selector import QuestionPairCandidate, QuestionPairSelector
from .question_subtype_classifier import QuestionSubtypeClassifier
from .question_type_classifier import QuestionTypeClassifier
from .schemas import ClassificationMeta, ClassificationResult, ThemePrediction
from .theme_selector import ThemeSelector
from .subtype_catalog import DEFAULT_SUBTYPE_CATALOG_PATH, load_subtype_catalog
from .text_preprocessor import (
    BasicTextPreprocessor,
    LLMTextPreprocessor,
    RetrievalQuery,
    TextPreprocessor,
)
from .validators import validate_classification_result


HIGH_CONFIDENCE_QUESTION_TYPE = 0.8
LOW_ALTERNATIVE_QUESTION_TYPE = 0.75


class ClassificationPipeline:
    def __init__(
        self,
        theme_retriever: HybridThemeRetriever,
        question_type_classifier: QuestionTypeClassifier,
        question_subtype_classifier: QuestionSubtypeClassifier,
        question_pair_selector: QuestionPairSelector,
        theme_selector: ThemeSelector,
        allowed_theme_codes: set[str],
        text_preprocessor: TextPreprocessor | None = None,
        theme_candidates_top_k: int = 30,
    ):
        self.theme_retriever = theme_retriever
        self.question_type_classifier = question_type_classifier
        self.question_subtype_classifier = question_subtype_classifier
        self.question_pair_selector = question_pair_selector
        self.theme_selector = theme_selector
        self.allowed_theme_codes = set(allowed_theme_codes)
        self.text_preprocessor = text_preprocessor or BasicTextPreprocessor()
        self.theme_candidates_top_k = max(1, theme_candidates_top_k)
        self.last_candidates: list[ThemeCandidate] = []
        self.last_preprocessed_text = ""
        self.last_normalized_text = ""
        self.last_type_decision_text = ""
        self.last_normalized_type_text = ""
        self.last_retrieval_queries: list[RetrievalQuery] = []
        self.last_retrieval_query_candidates: list[list[ThemeCandidate]] = []
        self.last_question_pair_candidates: list[QuestionPairCandidate] = []
        self.last_question_type_candidates = []
        self.last_theme_selector_diagnostics = None

    @staticmethod
    def _fallback_retrieval_queries(text: str) -> list[RetrievalQuery]:
        return [RetrievalQuery(section=None, query=text)] if text else []

    def _retrieval_queries(self, decision_text: str) -> list[RetrievalQuery]:
        queries = getattr(self.text_preprocessor, "last_retrieval_queries", None)
        if not isinstance(queries, list):
            return self._fallback_retrieval_queries(decision_text)
        if not queries:
            return self._fallback_retrieval_queries(decision_text)
        clean: list[RetrievalQuery] = []
        for query in queries[:4]:
            if isinstance(query, RetrievalQuery) and query.query.strip():
                clean.append(query)
        return clean or self._fallback_retrieval_queries(decision_text)

    def _search_theme_candidates(
        self, retrieval_queries: list[RetrievalQuery], top_k: int = 30
    ) -> list[ThemeCandidate]:
        merged: dict[str, dict[str, object]] = {}
        self.last_retrieval_query_candidates = []
        for retrieval_query in retrieval_queries:
            normalized_query = normalize_text(retrieval_query.query)
            if not normalized_query:
                self.last_retrieval_query_candidates.append([])
                continue
            candidates = self.theme_retriever.search(normalized_query, top_k=30)
            self.last_retrieval_query_candidates.append(candidates)
            for candidate in candidates:
                entry = merged.setdefault(
                    candidate.code,
                    {
                        "candidate": candidate,
                        "best_score": candidate.hybridScore,
                        "matched_count": 0,
                        "section_match": False,
                    },
                )
                entry["matched_count"] = int(entry["matched_count"]) + 1
                if candidate.hybridScore > float(entry["best_score"]):
                    entry["candidate"] = candidate
                    entry["best_score"] = candidate.hybridScore
                if retrieval_query.section and candidate.section == retrieval_query.section:
                    entry["section_match"] = True

        scored: list[tuple[float, str, ThemeCandidate]] = []
        for code, entry in merged.items():
            score = float(entry["best_score"])
            score += 0.03 * int(entry["matched_count"])
            if entry["section_match"]:
                score += 0.05
            score = max(0.0, score)
            scored.append((score, code, entry["candidate"]))  # type: ignore[arg-type]
        scored.sort(key=lambda item: (-item[0], item[1]))

        results: list[ThemeCandidate] = []
        for rank, (score, _, candidate) in enumerate(scored[:top_k], 1):
            results.append(
                candidate.model_copy(
                    update={
                        "hybridScore": score,
                        "rank": rank,
                        "source": "hybrid",
                    }
                )
            )
        return results

    def _question_pair_candidates(
        self, normalized_text: str, question_types
    ) -> list[QuestionPairCandidate]:
        if not question_types:
            return []
        top_type = question_types[0]
        if self._can_use_fast_question_type_path(question_types):
            subtypes = self.question_subtype_classifier.classify_top_k(
                normalized_text, top_type, k=2
            )
            return [
                QuestionPairCandidate(
                    question_type=top_type,
                    question_subtype=subtype,
                )
                for subtype in subtypes
            ]

        def classify_subtypes(question_type):
            return self.question_subtype_classifier.classify_top_k(
                normalized_text, question_type, k=2
            )

        pair_candidates: list[QuestionPairCandidate] = []
        with ThreadPoolExecutor(max_workers=len(question_types)) as executor:
            subtype_lists = list(executor.map(classify_subtypes, question_types))
        for question_type, subtypes in zip(question_types, subtype_lists):
            for subtype in subtypes:
                pair_candidates.append(
                    QuestionPairCandidate(
                        question_type=question_type,
                        question_subtype=subtype,
                    )
                )
        return pair_candidates

    @staticmethod
    def _can_use_fast_question_type_path(question_types) -> bool:
        if not question_types:
            return False
        top_type = question_types[0]
        second_confidence = question_types[1].confidence if len(question_types) > 1 else 0.0
        return (
            top_type.confidence >= HIGH_CONFIDENCE_QUESTION_TYPE
            and second_confidence <= LOW_ALTERNATIVE_QUESTION_TYPE
        )

    def run(self, text: str) -> ClassificationResult:
        raw_text = text or ""
        preprocessed = self.text_preprocessor.prepare(raw_text)
        self.last_preprocessed_text = preprocessed
        normalized = normalize_text(preprocessed)
        self.last_normalized_text = normalized
        type_decision_text = (
            getattr(self.text_preprocessor, "last_type_decision_text", "") or preprocessed
        )
        self.last_type_decision_text = type_decision_text
        normalized_type = normalize_text(type_decision_text)
        self.last_normalized_type_text = normalized_type
        self.last_retrieval_queries = self._retrieval_queries(preprocessed)
        candidates = self._search_theme_candidates(
            self.last_retrieval_queries, top_k=self.theme_candidates_top_k
        )
        self.last_candidates = candidates
        question_types = self.question_type_classifier.classify_top_k(
            normalized_type, k=2
        )
        self.last_question_type_candidates = question_types
        pair_candidates = self._question_pair_candidates(normalized_type, question_types)
        self.last_question_pair_candidates = pair_candidates
        if self._can_use_fast_question_type_path(question_types):
            selected_pair = pair_candidates[0]
        else:
            selected_pair = self.question_pair_selector.select(normalized_type, pair_candidates)
        question_type = selected_pair.question_type
        question_subtype = selected_pair.question_subtype
        selected = self.theme_selector.select(normalized, candidates, max_themes=3)
        self.last_theme_selector_diagnostics = getattr(
            self.theme_selector, "last_diagnostics", None
        )
        if not selected and candidates:
            top = candidates[0]
            selected = [
                ThemePrediction(
                    code=top.code,
                    name=top.name,
                    confidence=0.4,
                    section=top.section,
                )
            ]
        result = ClassificationResult(
            themes=selected,
            questionType=question_type,
            questionSubtype=question_subtype,
            meta=ClassificationMeta(
                requestId=uuid.uuid4().hex,
                processedAt=datetime.now(timezone.utc)
                .isoformat()
                .replace("+00:00", "Z"),
                sourceTextLength=len(raw_text),
            ),
        )
        errors = validate_classification_result(result, self.allowed_theme_codes)
        if errors:
            raise ClassificationValidationError(errors)
        return result


def build_pipeline(
    themes_path: Path,
    index_dir: Path,
    subtypes_path: Path = DEFAULT_SUBTYPE_CATALOG_PATH,
    theme_candidates_top_k: int = 30,
) -> ClassificationPipeline:
    themes = load_themes(themes_path)
    catalog = load_subtype_catalog(subtypes_path)
    if catalog.catalogVersion != themes[0].classifierVersion:
        raise ValueError("Theme and subtype classifier versions do not match")
    report = json.loads(
        (index_dir / "index_report.json").read_text(encoding="utf-8")
    )
    expected_hash = report.get("themesSha256")
    actual_hash = hashlib.sha256(themes_path.read_bytes()).hexdigest()
    if expected_hash != actual_hash:
        raise ValueError(
            "Theme indexes are stale or lack themesSha256; rebuild the index set"
        )
    metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    bm25 = BM25ThemeRetriever.load(index_dir / "bm25.pkl", themes)
    vector = FaissThemeRetriever.load(
        themes,
        index_dir / "faiss.index",
        index_dir / "embeddings.npy",
        index_dir / "metadata.json",
        metadata["embeddingModel"],
    )
    return ClassificationPipeline(
        HybridThemeRetriever(bm25, vector),
        QuestionTypeClassifier(),
        QuestionSubtypeClassifier(catalog=catalog),
        QuestionPairSelector(),
        ThemeSelector(),
        {theme.code for theme in themes},
        LLMTextPreprocessor(),
        theme_candidates_top_k=theme_candidates_top_k,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify a citizen request")
    parser.add_argument("--themes", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument(
        "--subtypes", type=Path, default=DEFAULT_SUBTYPE_CATALOG_PATH
    )
    parser.add_argument("--text", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    pipeline = build_pipeline(args.themes, args.index_dir, args.subtypes)
    result = pipeline.run(args.text)
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
