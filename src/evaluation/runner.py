from __future__ import annotations

import json
import time
from pathlib import Path

from src.classification.pipeline import ClassificationPipeline

from .metrics import calculate_metrics
from .schemas import (
    EvaluationDataset,
    PredictionRecord,
    QuestionPairDiagnostics,
    QuestionTypeDiagnostics,
    RetrievalDiagnostics,
    RetrievalQueryDiagnostics,
    TextDiagnostics,
)


def _fallback_used(record: PredictionRecord) -> bool:
    prediction = record.prediction
    return bool(
        prediction
        and (
            any(theme.confidence == 0.4 for theme in prediction.themes)
            or prediction.questionType.confidence == 0.3
            or prediction.questionSubtype.confidence == 0.3
        )
    )


class EvaluationRunner:
    def __init__(self, pipeline: ClassificationPipeline):
        self.pipeline = pipeline

    @staticmethod
    def load_records(path: Path) -> list[PredictionRecord]:
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(PredictionRecord.model_validate_json(line))
        return records

    def _retrieval_diagnostics(self, case) -> RetrievalDiagnostics | None:
        candidates = getattr(self.pipeline, "last_candidates", None)
        if candidates is None:
            return None
        ranks = {candidate.code: index for index, candidate in enumerate(candidates, 1)}
        return RetrievalDiagnostics(
            expectedThemeRanks={
                theme.code: ranks.get(theme.code) for theme in case.expected.themes
            },
            topCandidateCodes=[candidate.code for candidate in candidates[:10]],
        )

    def _text_diagnostics(self, source_text: str) -> TextDiagnostics:
        preprocessed = getattr(self.pipeline, "last_preprocessed_text", "") or ""
        normalized = getattr(self.pipeline, "last_normalized_text", "") or ""
        retrieval_queries = getattr(self.pipeline, "last_retrieval_queries", []) or []
        per_query_candidates = (
            getattr(self.pipeline, "last_retrieval_query_candidates", []) or []
        )
        query_diagnostics = []
        for index, query in enumerate(retrieval_queries):
            candidates = (
                per_query_candidates[index]
                if index < len(per_query_candidates)
                else []
            )
            query_diagnostics.append(
                RetrievalQueryDiagnostics(
                    section=getattr(query, "section", None),
                    query=getattr(query, "query", ""),
                    topCandidateCodes=[
                        candidate.code for candidate in candidates[:10]
                    ],
                )
            )
        return TextDiagnostics(
            sourceLength=len(source_text or ""),
            preprocessedLength=len(preprocessed),
            normalizedLength=len(normalized),
            preprocessedText=preprocessed[:2000],
            normalizedText=normalized[:2000],
            typeDecisionText=(getattr(self.pipeline, "last_type_decision_text", "") or "")[:2000],
            retrievalQueries=query_diagnostics,
        )

    def _question_type_diagnostics(self) -> list[QuestionTypeDiagnostics]:
        candidates = getattr(self.pipeline, "last_question_type_candidates", []) or []
        diagnostics: list[QuestionTypeDiagnostics] = []
        for candidate in candidates:
            diagnostics.append(
                QuestionTypeDiagnostics(
                    code=candidate.code,
                    name=candidate.name,
                    confidence=candidate.confidence,
                    reason=getattr(candidate, "reason", "") or "",
                )
            )
        return diagnostics

    def _question_pair_diagnostics(self) -> list[QuestionPairDiagnostics]:
        pair_candidates = (
            getattr(self.pipeline, "last_question_pair_candidates", []) or []
        )
        diagnostics: list[QuestionPairDiagnostics] = []
        for candidate in pair_candidates:
            question_type = candidate.question_type
            subtype = candidate.question_subtype
            diagnostics.append(
                QuestionPairDiagnostics(
                    key=candidate.key,
                    questionTypeCode=question_type.code,
                    questionTypeName=question_type.name,
                    questionSubtypeCode=subtype.code,
                    questionSubtypeOfficialCode=subtype.officialCode,
                    questionSubtypeName=subtype.name,
                )
            )
        return diagnostics

    def run(
        self,
        dataset: EvaluationDataset,
        out_dir: Path,
        resume: bool = False,
        metrics_only: bool = False,
    ) -> dict:
        out_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = out_dir / "predictions.jsonl"
        records = self.load_records(predictions_path)
        if not metrics_only:
            if records and not resume:
                raise FileExistsError(
                    f"{predictions_path} exists; use --resume or a new output directory"
                )
            completed = {record.id for record in records}
            with predictions_path.open("a", encoding="utf-8") as stream:
                for case in dataset.cases:
                    if case.id in completed:
                        continue
                    started = time.perf_counter()
                    try:
                        prediction = self.pipeline.run(case.text)
                        retrieval = self._retrieval_diagnostics(case)
                        record = PredictionRecord(
                            id=case.id,
                            prediction=prediction,
                            latencyMs=(time.perf_counter() - started) * 1000,
                            retrieval=retrieval,
                            textDiagnostics=self._text_diagnostics(case.text),
                            questionTypeCandidates=self._question_type_diagnostics(),
                            questionPairCandidates=self._question_pair_diagnostics(),
                        )
                        record.fallbackUsed = _fallback_used(record)
                    except Exception as error:
                        retrieval = self._retrieval_diagnostics(case)
                        record = PredictionRecord(
                            id=case.id,
                            latencyMs=(time.perf_counter() - started) * 1000,
                            error=f"{type(error).__name__}: {error}",
                            retrieval=retrieval,
                        )
                    stream.write(record.model_dump_json() + "\n")
                    stream.flush()
                    records.append(record)
        return calculate_metrics(dataset, records, out_dir)
