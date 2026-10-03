from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="With --resume, retry records that previously ended with an error.",
    )
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument(
        "--llm-min-interval-seconds",
        type=float,
        default=None,
        help="Minimum delay between LLM calls in this process; useful for low RPM API limits.",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=None,
        help="How many times to retry 429/rate-limit LLM errors.",
    )
    parser.add_argument(
        "--llm-retry-seconds",
        type=float,
        default=None,
        help="Delay before retrying after a 429/rate-limit LLM error.",
    )
    parser.add_argument(
        "--theme-candidates-top-k",
        type=int,
        default=30,
        help="How many merged theme candidates to pass to ThemeSelector.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.llm_min_interval_seconds is not None:
        os.environ["HYDRA_LLM_MIN_INTERVAL_SECONDS"] = str(
            args.llm_min_interval_seconds
        )
    if args.llm_max_retries is not None:
        os.environ["HYDRA_LLM_MAX_RETRIES"] = str(args.llm_max_retries)
    if args.llm_retry_seconds is not None:
        os.environ["HYDRA_LLM_RETRY_SECONDS"] = str(args.llm_retry_seconds)
    themes = load_themes(args.themes)
    catalog = load_subtype_catalog(args.subtypes)
    dataset = load_evaluation_dataset(args.dataset, themes, catalog)
    pipeline = build_pipeline(
        args.themes,
        args.index_dir,
        args.subtypes,
        theme_candidates_top_k=args.theme_candidates_top_k,
    )
    report = EvaluationRunner(pipeline).run(
        dataset, args.out_dir, args.resume, args.metrics_only, args.retry_failed
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
