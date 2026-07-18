from pathlib import Path

from src.classification.schemas import (
    ClassificationMeta,
    ClassificationResult,
    QuestionSubtypePrediction,
    QuestionTypePrediction,
    ThemePrediction,
)
from src.classification.subtype_catalog import load_subtype_catalog
from src.evaluation.loader import load_evaluation_dataset
from src.evaluation.metrics import calculate_metrics
from src.evaluation.runner import EvaluationRunner
from src.evaluation.schemas import PredictionRecord
from src.evaluation.schemas import RetrievalDiagnostics
from src.evaluation import run as run_module
from src.retrieval.hybrid_theme_retriever import load_themes

OUTPUT = Path("data/evaluation/test-output")


def clean_output():
    for path in OUTPUT.iterdir():
        if path.name != ".gitkeep" and path.is_file():
            path.unlink()


def seed():
    themes = load_themes(Path("data/classifiers/themes_leaf.json"))
    catalog = load_subtype_catalog()
    dataset = load_evaluation_dataset(
        Path("data/evaluation/seed_v1.json"), themes, catalog
    )
    return dataset


def prediction_for(case):
    subtype = case.expected.questionSubtype
    return ClassificationResult(
        themes=[
            ThemePrediction(
                code=theme.code,
                name=theme.name,
                section=theme.section,
                confidence=0.9,
            )
            for theme in case.expected.themes
        ],
        questionType=QuestionTypePrediction(
            code=case.expected.questionType.code,
            name=case.expected.questionType.name,
            confidence=0.9,
        ),
        questionSubtype=QuestionSubtypePrediction(
            code=subtype.code,
            officialCode=subtype.officialCode,
            name=subtype.name,
            confidence=0.9,
        ),
        meta=ClassificationMeta(
            requestId=case.id,
            processedAt="2026-07-06T00:00:00Z",
            sourceTextLength=len(case.text),
        ),
    )


def test_seed_dataset_is_valid():
    dataset = seed()
    assert len(dataset.cases) == 50
    assert len({case.id for case in dataset.cases}) == len(dataset.cases)


def test_metrics_are_exact_for_perfect_predictions():
    dataset = seed()
    records = [
        PredictionRecord(
            id=case.id,
            prediction=prediction_for(case),
            latencyMs=10,
            retrieval=RetrievalDiagnostics(
                expectedThemeRanks={theme.code: 1 for theme in case.expected.themes},
                topCandidateCodes=[theme.code for theme in case.expected.themes],
            ),
        )
        for case in dataset.cases
    ]
    clean_output()
    report = calculate_metrics(dataset, records, OUTPUT)
    assert report["themes"]["microF1"] == 1
    assert report["themeSelection"]["expectedThemeIncludedRate"] == 1
    assert report["retrieval"]["expectedInTop1"] == 1
    assert report["retrieval"]["expectedInTop20"] == 1
    assert report["retrieval"]["expectedInTop30"] == 1
    assert (OUTPUT / "retrieval_diagnostics.csv").exists()
    assert report["questionType"]["accuracy"] == 1
    assert (OUTPUT / "question_type_confusion.csv").exists()
    assert (OUTPUT / "theme_per_label_metrics.csv").exists()
    clean_output()


def test_runner_resumes_and_records_partial_failure():
    dataset = seed()

    class Pipeline:
        def run(self, text):
            case = next(case for case in dataset.cases if case.text == text)
            if case.id == "seed-002":
                raise RuntimeError("provider failed")
            return prediction_for(case)

    clean_output()
    runner = EvaluationRunner(Pipeline())
    first = runner.run(dataset, OUTPUT)
    assert first["failed"] == 1
    lines_before = (OUTPUT / "predictions.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    second = runner.run(dataset, OUTPUT, resume=True)
    lines_after = (OUTPUT / "predictions.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines_before) == len(lines_after) == len(dataset.cases)
    assert second["processed"] == len(dataset.cases)
    clean_output()


def test_evaluation_cli_coordinates_components(monkeypatch, capsys):
    dataset = seed()
    arguments = type(
        "Arguments",
        (),
        {
            "dataset": Path("dataset.json"),
            "themes": Path("themes.json"),
            "subtypes": Path("subtypes.json"),
            "index_dir": Path("indexes"),
            "out_dir": Path("output"),
            "resume": True,
            "metrics_only": False,
        },
    )()

    class Parser:
        def parse_args(self):
            return arguments

    class Runner:
        def __init__(self, pipeline):
            assert pipeline == "pipeline"

        def run(self, loaded, out_dir, resume, metrics_only):
            assert loaded is dataset
            assert resume is True
            return {"successful": 5}

    monkeypatch.setattr(run_module, "_parser", lambda: Parser())
    monkeypatch.setattr(run_module, "load_themes", lambda _: ["themes"])
    monkeypatch.setattr(run_module, "load_subtype_catalog", lambda _: "catalog")
    monkeypatch.setattr(
        run_module, "load_evaluation_dataset", lambda *args: dataset
    )
    monkeypatch.setattr(run_module, "build_pipeline", lambda *args: "pipeline")
    monkeypatch.setattr(run_module, "EvaluationRunner", Runner)
    assert run_module.main() == 0
    assert '"successful": 5' in capsys.readouterr().out
