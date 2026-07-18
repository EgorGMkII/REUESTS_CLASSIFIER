from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.classification.pipeline import build_pipeline
from src.classification.subtype_catalog import load_subtype_catalog
from src.retrieval.hybrid_theme_retriever import load_themes

from .loader import load_evaluation_dataset
from .runner import EvaluationRunner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate classification quality")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--themes", type=Path, required=True)
    parser.add_argument("--subtypes", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--metrics-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    themes = load_themes(args.themes)
    catalog = load_subtype_catalog(args.subtypes)
    dataset = load_evaluation_dataset(args.dataset, themes, catalog)
    pipeline = build_pipeline(args.themes, args.index_dir, args.subtypes)
    report = EvaluationRunner(pipeline).run(
        dataset, args.out_dir, args.resume, args.metrics_only
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
