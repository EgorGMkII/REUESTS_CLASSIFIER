from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    jaccard_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

from .schemas import EvaluationDataset, PredictionRecord


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _join_codes(items) -> str:
    return " ".join(item.code for item in items)


def _join_names(items) -> str:
    return " | ".join(f"{item.code}: {item.name}" for item in items)


def _join_theme_selector_items(items) -> str:
    return " | ".join(
        f"{item.code} {item.role}/{item.matchQuality}: {item.reason}"
        for item in items
    )


def _theme_error_kind(
    expected_codes: set[str],
    predicted_codes: set[str],
    expected_ranks: dict[str, int | None],
) -> str:
    if expected_codes == predicted_codes:
        return "exact_match"
    if expected_codes and expected_codes.issubset(predicted_codes):
        return "correct_with_extra"
    if expected_codes.intersection(predicted_codes):
        return "partial_match"
    known_ranks = [rank for rank in expected_ranks.values() if rank is not None]
    if known_ranks:
        return "selector_missed_candidate"
    return "retrieval_missed_expected"


def _write_classification_diagnostics(
    dataset: EvaluationDataset,
    records: list[PredictionRecord],
    out_dir: Path,
) -> None:
    cases = {case.id: case for case in dataset.cases}
    rows: list[list[object]] = []
    markdown: list[str] = [
        "# Classification diagnostics",
        "",
        "Короткий отчёт по каждому кейсу: ожидание, предсказание, ранг ожидаемой темы в retrieval и тип ошибки.",
        "",
    ]
    for record in records:
        case = cases.get(record.id)
        if case is None:
            continue
        expected = case.expected
        prediction = record.prediction
        expected_theme_codes = {theme.code for theme in expected.themes}
        expected_ranks = (
            record.retrieval.expectedThemeRanks
            if record.retrieval is not None
            else {}
        )
        known_ranks = [rank for rank in expected_ranks.values() if rank is not None]
        best_rank = min(known_ranks) if known_ranks else None
        if prediction is None:
            rows.append(
                [
                    record.id,
                    "failed",
                    _join_codes(expected.themes),
                    "",
                    "" if best_rank is None else best_rank,
                    expected.questionType.code,
                    "",
                    expected.questionSubtype.code,
                    "",
                    "no",
                    "no",
                    record.error or "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )
            continue

        predicted_theme_codes = {theme.code for theme in prediction.themes}
        kind = _theme_error_kind(
            expected_theme_codes, predicted_theme_codes, expected_ranks
        )
        type_match = expected.questionType.code == prediction.questionType.code
        subtype_match = expected.questionSubtype.code == prediction.questionSubtype.code
        selected_expected = bool(expected_theme_codes.intersection(predicted_theme_codes))
        text_preview = ""
        if record.textDiagnostics is not None:
            text_preview = (
                record.textDiagnostics.preprocessedText
                or record.textDiagnostics.normalizedText
            )[:500].replace("\n", " ")
        pair_candidates = " ".join(
            candidate.key for candidate in record.questionPairCandidates
        )
        theme_selected_diagnostics = ""
        theme_rejected_diagnostics = ""
        if record.themeSelectorDiagnostics is not None:
            theme_selected_diagnostics = _join_theme_selector_items(
                record.themeSelectorDiagnostics.selected
            )
            theme_rejected_diagnostics = _join_theme_selector_items(
                record.themeSelectorDiagnostics.rejected
            )
        type_candidates = " | ".join(
            f"{candidate.code} {candidate.name} {candidate.confidence:.2f}: {candidate.reason}"
            for candidate in record.questionTypeCandidates
        )
        type_decision_preview = ""
        if record.textDiagnostics is not None:
            type_decision_preview = record.textDiagnostics.typeDecisionText[:500].replace("\n", " ")
        rows.append(
            [
                record.id,
                kind,
                _join_codes(expected.themes),
                _join_codes(prediction.themes),
                "" if best_rank is None else best_rank,
                expected.questionType.code,
                prediction.questionType.code,
                expected.questionSubtype.code,
                prediction.questionSubtype.code,
                "yes" if type_match else "no",
                "yes" if subtype_match else "no",
                record.error or "",
                text_preview,
                type_decision_preview,
                type_candidates,
                pair_candidates,
                theme_selected_diagnostics,
                theme_rejected_diagnostics,
            ]
        )

        markdown.extend(
            [
                f"## {record.id}",
                "",
                f"- Тема: `{kind}`; лучший rank expected: `{best_rank if best_rank is not None else 'not_found'}`; expected selected: `{'yes' if selected_expected else 'no'}`.",
                f"- Expected themes: {_join_names(expected.themes)}",
                f"- Predicted themes: {_join_names(prediction.themes)}",
                f"- Type: expected `{expected.questionType.code}` {expected.questionType.name}; predicted `{prediction.questionType.code}` {prediction.questionType.name}; match: `{'yes' if type_match else 'no'}`.",
                f"- Subtype: expected `{expected.questionSubtype.code}` / {expected.questionSubtype.officialCode}; predicted `{prediction.questionSubtype.code}` / {prediction.questionSubtype.officialCode}; match: `{'yes' if subtype_match else 'no'}`.",
            ]
        )
        if pair_candidates:
            markdown.append(f"- Pair candidates: `{pair_candidates}`")
        if theme_selected_diagnostics:
            markdown.append(f"- Theme selector selected: {theme_selected_diagnostics}")
        if theme_rejected_diagnostics:
            markdown.append(f"- Theme selector rejected: {theme_rejected_diagnostics}")
        if type_candidates:
            markdown.append(f"- Type candidates: {type_candidates}")
        if text_preview:
            markdown.append(f"- Decision text: {text_preview}")
        if type_decision_preview and type_decision_preview != text_preview:
            markdown.append(f"- Type decision text: {type_decision_preview}")
        markdown.append("")

    _write_csv(
        out_dir / "classification_diagnostics.csv",
        [
            "id",
            "themeErrorKind",
            "expectedThemeCodes",
            "predictedThemeCodes",
            "bestExpectedThemeRank",
            "expectedQuestionType",
            "predictedQuestionType",
            "expectedQuestionSubtype",
            "predictedQuestionSubtype",
            "questionTypeMatch",
            "questionSubtypeMatch",
            "error",
            "decisionTextPreview",
            "typeDecisionTextPreview",
            "questionTypeCandidates",
            "questionPairCandidates",
            "themeSelectorSelected",
            "themeSelectorRejected",
        ],
        rows,
    )
    (out_dir / "classification_diagnostics.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )


def _plot_confusion(path: Path, matrix: np.ndarray, labels: list[str], title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        size = max(6, min(16, len(labels) * 0.7))
        figure, axis = plt.subplots(figsize=(size, size))
        image = axis.imshow(matrix, cmap="Blues")
        figure.colorbar(image, ax=axis)
        axis.set(
            xticks=range(len(labels)),
            yticks=range(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
            xlabel="Predicted",
            ylabel="Expected",
            title=title,
        )
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
    except Exception:
        return


def calculate_metrics(
    dataset: EvaluationDataset,
    records: list[PredictionRecord],
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_classification_diagnostics(dataset, records, out_dir)
    cases = {case.id: case for case in dataset.cases}
    successful = [record for record in records if record.prediction is not None and record.id in cases]
    expected_themes = [
        [theme.code for theme in cases[record.id].expected.themes] for record in successful
    ]
    predicted_themes = [
        [theme.code for theme in record.prediction.themes] for record in successful
    ]
    all_theme_labels = sorted(
        {code for values in expected_themes + predicted_themes for code in values}
    )
    if successful and all_theme_labels:
        binarizer = MultiLabelBinarizer(classes=all_theme_labels)
        binarizer.fit([all_theme_labels])
        expected_binary = binarizer.transform(expected_themes)
        predicted_binary = binarizer.transform(predicted_themes)
        theme_metrics = {
            "subsetAccuracy": float(accuracy_score(expected_binary, predicted_binary)),
            "jaccardSamples": float(
                jaccard_score(expected_binary, predicted_binary, average="samples", zero_division=0)
            ),
            "microPrecision": float(precision_score(expected_binary, predicted_binary, average="micro", zero_division=0)),
            "microRecall": float(recall_score(expected_binary, predicted_binary, average="micro", zero_division=0)),
            "microF1": float(f1_score(expected_binary, predicted_binary, average="micro", zero_division=0)),
            "macroPrecision": float(precision_score(expected_binary, predicted_binary, average="macro", zero_division=0)),
            "macroRecall": float(recall_score(expected_binary, predicted_binary, average="macro", zero_division=0)),
            "macroF1": float(f1_score(expected_binary, predicted_binary, average="macro", zero_division=0)),
        }
        per_label = multilabel_confusion_matrix(expected_binary, predicted_binary)
        rows = []
        for label, matrix in zip(all_theme_labels, per_label):
            tn, fp, fn, tp = matrix.ravel()
            rows.append([label, int(tp), int(fp), int(fn), int(tn)])
        _write_csv(out_dir / "theme_per_label_metrics.csv", ["code", "tp", "fp", "fn", "tn"], rows)
    else:
        theme_metrics = {name: 0.0 for name in (
            "subsetAccuracy", "jaccardSamples", "microPrecision", "microRecall",
            "microF1", "macroPrecision", "macroRecall", "macroF1"
        )}
        _write_csv(out_dir / "theme_per_label_metrics.csv", ["code", "tp", "fp", "fn", "tn"], [])

    substitutions: Counter[tuple[str, str]] = Counter()
    correct_with_extra = 0
    theme_hit_cases = 0
    theme_hit_at_1 = 0
    theme_hit_at_3 = 0
    for expected, predicted in zip(expected_themes, predicted_themes):
        expected_set = set(expected)
        predicted_set = set(predicted)
        if expected_set and expected_set.issubset(predicted_set):
            theme_hit_cases += 1
            if predicted_set != expected_set:
                correct_with_extra += 1
        if predicted[:1] and expected_set.intersection(predicted[:1]):
            theme_hit_at_1 += 1
        if predicted[:3] and expected_set.intersection(predicted[:3]):
            theme_hit_at_3 += 1
        missing = expected_set - predicted_set
        extra = predicted_set - expected_set
        for expected_code in missing:
            for predicted_code in extra:
                substitutions[(expected_code, predicted_code)] += 1
    _write_csv(
        out_dir / "theme_top_confusions.csv",
        ["expectedCode", "predictedCode", "count"],
        [[left, right, count] for (left, right), count in substitutions.most_common()],
    )

    retrieval_rows: list[list[object]] = []
    retrieval_rank_values: list[int] = []
    retrieval_hit_at_1 = 0
    retrieval_hit_at_3 = 0
    retrieval_hit_at_10 = 0
    retrieval_hit_at_20 = 0
    retrieval_hit_at_30 = 0
    retrieval_hit_at_50 = 0
    retrieval_records = 0
    for record in records:
        if record.id not in cases or record.retrieval is None:
            continue
        retrieval_records += 1
        case = cases[record.id]
        expected_codes = [theme.code for theme in case.expected.themes]
        ranks = [
            record.retrieval.expectedThemeRanks.get(code) for code in expected_codes
        ]
        known_ranks = [rank for rank in ranks if rank is not None]
        best_rank = min(known_ranks) if known_ranks else None
        if best_rank is not None:
            retrieval_rank_values.append(best_rank)
            retrieval_hit_at_1 += int(best_rank <= 1)
            retrieval_hit_at_3 += int(best_rank <= 3)
            retrieval_hit_at_10 += int(best_rank <= 10)
            retrieval_hit_at_20 += int(best_rank <= 20)
            retrieval_hit_at_30 += int(best_rank <= 30)
            retrieval_hit_at_50 += int(best_rank <= 50)
        retrieval_rows.append(
            [
                record.id,
                " ".join(expected_codes),
                "" if best_rank is None else best_rank,
                " ".join(record.retrieval.topCandidateCodes),
            ]
        )
    _write_csv(
        out_dir / "retrieval_diagnostics.csv",
        ["id", "expectedThemeCodes", "bestExpectedRank", "topCandidateCodes"],
        retrieval_rows,
    )

    expected_types = [cases[r.id].expected.questionType.code for r in successful]
    predicted_types = [r.prediction.questionType.code for r in successful]
    type_labels = [str(number) for number in range(1, 10)]
    type_matrix = confusion_matrix(expected_types, predicted_types, labels=type_labels)
    _write_csv(
        out_dir / "question_type_confusion.csv",
        ["expected\\predicted", *type_labels],
        [[label, *map(int, row)] for label, row in zip(type_labels, type_matrix)],
    )
    _plot_confusion(out_dir / "question_type_confusion.png", type_matrix, type_labels, "Question type")

    expected_subtypes = [cases[r.id].expected.questionSubtype.code for r in successful]
    predicted_subtypes = [r.prediction.questionSubtype.code for r in successful]
    subtype_labels = sorted(set(expected_subtypes + predicted_subtypes))
    subtype_matrix = confusion_matrix(expected_subtypes, predicted_subtypes, labels=subtype_labels)
    _write_csv(
        out_dir / "question_subtype_confusion.csv",
        ["expected\\predicted", *subtype_labels],
        [[label, *map(int, row)] for label, row in zip(subtype_labels, subtype_matrix)],
    )
    _plot_confusion(out_dir / "question_subtype_confusion.png", subtype_matrix, subtype_labels, "Question subtype")

    latencies = [record.latencyMs for record in records]
    report = {
        "datasetVersion": dataset.datasetVersion,
        "classifierVersion": dataset.classifierVersion,
        "total": len(dataset.cases),
        "processed": len(records),
        "successful": len(successful),
        "failed": sum(record.prediction is None for record in records),
        "fallbackCount": sum(record.fallbackUsed for record in records),
        "latencyMs": {
            "p50": float(np.percentile(latencies, 50)) if latencies else 0.0,
            "p95": float(np.percentile(latencies, 95)) if latencies else 0.0,
        },
        "themes": theme_metrics,
        "themeSelection": {
            "expectedThemeIncludedRate": float(theme_hit_cases / len(successful)) if successful else 0.0,
            "hitAt1": float(theme_hit_at_1 / len(successful)) if successful else 0.0,
            "hitAt3": float(theme_hit_at_3 / len(successful)) if successful else 0.0,
            "correctWithExtraCount": correct_with_extra,
            "avgPredictedThemes": float(np.mean([len(values) for values in predicted_themes])) if predicted_themes else 0.0,
        },
        "retrieval": {
            "recordsWithDiagnostics": retrieval_records,
            "expectedInTop1": float(retrieval_hit_at_1 / retrieval_records) if retrieval_records else 0.0,
            "expectedInTop3": float(retrieval_hit_at_3 / retrieval_records) if retrieval_records else 0.0,
            "expectedInTop10": float(retrieval_hit_at_10 / retrieval_records) if retrieval_records else 0.0,
            "expectedInTop20": float(retrieval_hit_at_20 / retrieval_records) if retrieval_records else 0.0,
            "expectedInTop30": float(retrieval_hit_at_30 / retrieval_records) if retrieval_records else 0.0,
            "expectedInTop50": float(retrieval_hit_at_50 / retrieval_records) if retrieval_records else 0.0,
            "meanBestRank": float(np.mean(retrieval_rank_values)) if retrieval_rank_values else 0.0,
            "mrr": float(np.mean([1 / rank for rank in retrieval_rank_values])) if retrieval_rank_values else 0.0,
        },
        "questionType": {
            "accuracy": float(accuracy_score(expected_types, predicted_types)) if successful else 0.0,
            "macroF1": float(f1_score(expected_types, predicted_types, labels=type_labels, average="macro", zero_division=0)) if successful else 0.0,
        },
        "questionSubtype": {
            "accuracy": float(accuracy_score(expected_subtypes, predicted_subtypes)) if successful else 0.0,
            "macroF1": float(f1_score(expected_subtypes, predicted_subtypes, labels=subtype_labels, average="macro", zero_division=0)) if successful and subtype_labels else 0.0,
        },
    }
    (out_dir / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
