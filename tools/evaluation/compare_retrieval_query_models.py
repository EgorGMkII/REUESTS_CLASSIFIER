from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATASET = PROJECT_ROOT / "data" / "evaluation" / "real_v3.json"
DEFAULT_THEMES = PROJECT_ROOT / "data" / "classifiers" / "themes_leaf.json"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "indexes" / "themes"
DEFAULT_OUT_DIR = (
    PROJECT_ROOT / "data" / "evaluation" / "retrieval_query_model_compare" / "real-v3"
)
DEFAULT_MODELS = ["gpt-5-mini", "gpt-5.4-mini"]
TOP_K_VALUES = [1, 3, 10, 20, 30, 50]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare retrieval query generation quality across LLM models"
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--themes", type=Path, default=DEFAULT_THEMES)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--query-top-k", type=int, default=30)
    parser.add_argument("--final-top-k", type=int, default=50)
    parser.add_argument("--bm25-weight", type=float, default=0.45)
    parser.add_argument("--vector-weight", type=float, default=0.55)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-id", action="append", default=None)
    return parser


class ChatCompletionCallable:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        timeout: int,
        max_tokens: int,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.last_usage: dict[str, int | None] = {
            "promptTokens": None,
            "completionTokens": None,
            "totalTokens": None,
        }
        self.last_error: str | None = None

    def __call__(self, prompt: str) -> str:
        self.last_error = None
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            raise
        try:
            data = response.json()
        except ValueError:
            data = {"rawText": response.text[:2000]}

        if response.status_code >= 400:
            self.last_error = f"LLM request failed: {response.status_code} {data}"
            raise RuntimeError(self.last_error)

        self.last_usage = extract_usage(data)
        try:
            return extract_chat_text(data)
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            raise


def extract_chat_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise ValueError("LLM response is not a JSON object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response has no choices")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        raise ValueError("LLM response choice has no message")
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    raise ValueError("LLM response message content is not text")


def extract_usage(payload: Any) -> dict[str, int | None]:
    if not isinstance(payload, dict) or not isinstance(payload.get("usage"), dict):
        return {"promptTokens": None, "completionTokens": None, "totalTokens": None}
    usage = payload["usage"]
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    return {
        "promptTokens": prompt_tokens if isinstance(prompt_tokens, int) else None,
        "completionTokens": completion_tokens if isinstance(completion_tokens, int) else None,
        "totalTokens": total_tokens if isinstance(total_tokens, int) else None,
    }


def build_retriever(
    themes_path: Path,
    index_dir: Path,
    bm25_weight: float,
    vector_weight: float,
):
    from src.retrieval.bm25_retriever import BM25ThemeRetriever
    from src.retrieval.faiss_retriever import FaissThemeRetriever
    from src.retrieval.hybrid_theme_retriever import HybridThemeRetriever, load_themes

    themes = load_themes(themes_path)
    metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    bm25 = BM25ThemeRetriever.load(index_dir / "bm25.pkl", themes)
    vector = FaissThemeRetriever.load(
        themes,
        index_dir / "faiss.index",
        index_dir / "embeddings.npy",
        index_dir / "metadata.json",
        metadata["embeddingModel"],
    )
    return HybridThemeRetriever(
        bm25,
        vector,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
    )


def search_and_merge(
    retriever,
    retrieval_queries: list[Any],
    query_top_k: int,
    final_top_k: int,
) -> tuple[list[Any], list[list[Any]]]:
    from src.retrieval.text_normalization import normalize_text

    merged: dict[str, dict[str, object]] = {}
    per_query_candidates: list[list[Any]] = []

    for retrieval_query in retrieval_queries:
        normalized_query = normalize_text(retrieval_query.query)
        if not normalized_query:
            per_query_candidates.append([])
            continue
        candidates = retriever.search(normalized_query, top_k=query_top_k)
        per_query_candidates.append(candidates)
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

    scored: list[tuple[float, str, Any]] = []
    for code, entry in merged.items():
        score = float(entry["best_score"])
        score += 0.03 * int(entry["matched_count"])
        if entry["section_match"]:
            score += 0.05
        scored.append((max(0.0, score), code, entry["candidate"]))  # type: ignore[arg-type]
    scored.sort(key=lambda item: (-item[0], item[1]))

    merged_candidates: list[Any] = []
    for rank, (score, _, candidate) in enumerate(scored[:final_top_k], start=1):
        merged_candidates.append(
            candidate.model_copy(
                update={"hybridScore": score, "rank": rank, "source": "hybrid"}
            )
        )
    return merged_candidates, per_query_candidates


def expected_ranks(expected_codes: list[str], candidates: list[Any]) -> dict[str, int | None]:
    ranks_by_code = {candidate.code: candidate.rank for candidate in candidates}
    return {code: ranks_by_code.get(code) for code in expected_codes}


def best_rank(ranks: dict[str, int | None]) -> int | None:
    values = [rank for rank in ranks.values() if isinstance(rank, int)]
    return min(values) if values else None


def query_to_dict(query: Any) -> dict[str, str | None]:
    return {"section": query.section, "query": query.query}


def candidate_to_short_dict(candidate: Any) -> dict[str, Any]:
    return {
        "rank": candidate.rank,
        "code": candidate.code,
        "name": candidate.name,
        "section": candidate.section,
        "hybridScore": candidate.hybridScore,
    }


def make_record(
    case: dict[str, Any],
    model: str,
    decision_text: str,
    decomposition: dict[str, Any],
    retrieval_queries: list[Any],
    candidates: list[Any],
    per_query_candidates: list[list[Any]],
    usage: dict[str, int | None],
    error: str | None = None,
) -> dict[str, Any]:
    expected_themes = case["expected"]["themes"]
    expected_codes = [theme["code"] for theme in expected_themes]
    ranks = expected_ranks(expected_codes, candidates)
    return {
        "id": case["id"],
        "model": model,
        "decisionText": decision_text,
        "anchorId": decomposition.get("anchorId") or "",
        "mainSubject": decomposition.get("mainSubject") or "",
        "domain": decomposition.get("domain") or "",
        "primaryProblem": decomposition.get("primaryProblem") or "",
        "secondaryProblems": decomposition.get("secondaryProblems") or [],
        "detailsToDrop": decomposition.get("detailsToDrop") or [],
        "retrievalQueries": [query_to_dict(query) for query in retrieval_queries],
        "usage": usage,
        "expectedThemes": [
            {
                "code": theme["code"],
                "name": theme["name"],
                "section": theme["section"],
            }
            for theme in expected_themes
        ],
        "topCandidateCodes": [candidate.code for candidate in candidates],
        "topCandidates": [candidate_to_short_dict(candidate) for candidate in candidates[:10]],
        "perQueryTopCandidateCodes": [
            [candidate.code for candidate in query_candidates[:10]]
            for query_candidates in per_query_candidates
        ],
        "expectedThemeRanks": ranks,
        "bestRank": best_rank(ranks),
        "error": error,
    }


def make_error_record(case: dict[str, Any], model: str, error: Exception) -> dict[str, Any]:
    return make_record(
        case=case,
        model=model,
        decision_text="",
        decomposition={},
        retrieval_queries=[],
        candidates=[],
        per_query_candidates=[],
        usage={"promptTokens": None, "completionTokens": None, "totalTokens": None},
        error=f"{type(error).__name__}: {error}",
    )


def calculate_report(records: list[dict[str, Any]], models: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for model in models:
        model_records = [record for record in records if record["model"] == model]
        successful = [record for record in model_records if not record.get("error")]
        ranks = [
            record["bestRank"]
            for record in successful
            if isinstance(record.get("bestRank"), int)
        ]
        metrics = {
            "total": len(model_records),
            "successful": len(successful),
            "failed": len(model_records) - len(successful),
            "recordsWithDiagnostics": len(successful),
            "usage": sum_usage(model_records),
        }
        for top_k in TOP_K_VALUES:
            hits = sum(1 for rank in ranks if rank <= top_k)
            metrics[f"expectedInTop{top_k}"] = (
                hits / len(successful) if successful else 0.0
            )
        metrics["meanBestRank"] = sum(ranks) / len(ranks) if ranks else 0.0
        metrics["mrr"] = sum(1 / rank for rank in ranks) / len(ranks) if ranks else 0.0
        report[model] = metrics
    return report


def sum_usage(records: list[dict[str, Any]]) -> dict[str, int | None]:
    totals: dict[str, int | None] = {
        "promptTokens": 0,
        "completionTokens": 0,
        "totalTokens": 0,
    }
    has_usage = False
    for record in records:
        usage = record.get("usage")
        if not isinstance(usage, dict):
            continue
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] = int(totals[key] or 0) + value
                has_usage = True
    if not has_usage:
        return {"promptTokens": None, "completionTokens": None, "totalTokens": None}
    return totals


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "model",
        "expected_codes",
        "best_rank",
        "hit_top_1",
        "hit_top_3",
        "hit_top_10",
        "hit_top_20",
        "hit_top_30",
        "hit_top_50",
        "queries",
        "anchor_id",
        "main_subject",
        "primary_problem",
        "details_to_drop",
        "top_10_codes",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            rank = record.get("bestRank")
            writer.writerow(
                {
                    "id": record["id"],
                    "model": record["model"],
                    "expected_codes": " ".join(
                        theme["code"] for theme in record["expectedThemes"]
                    ),
                    "best_rank": rank if rank is not None else "",
                    "hit_top_1": int(isinstance(rank, int) and rank <= 1),
                    "hit_top_3": int(isinstance(rank, int) and rank <= 3),
                    "hit_top_10": int(isinstance(rank, int) and rank <= 10),
                    "hit_top_20": int(isinstance(rank, int) and rank <= 20),
                    "hit_top_30": int(isinstance(rank, int) and rank <= 30),
                    "hit_top_50": int(isinstance(rank, int) and rank <= 50),
                    "queries": " | ".join(
                        f"{query.get('section') or '-'}: {query.get('query')}"
                        for query in record["retrievalQueries"]
                    ),
                    "anchor_id": record.get("anchorId") or "",
                    "main_subject": record.get("mainSubject") or "",
                    "primary_problem": record.get("primaryProblem") or "",
                    "details_to_drop": " | ".join(record.get("detailsToDrop") or []),
                    "top_10_codes": " ".join(record["topCandidateCodes"][:10]),
                    "error": record.get("error") or "",
                }
            )


def main() -> int:
    from src.classification.text_preprocessor import LLMTextPreprocessor

    args = _parser().parse_args()
    api_key = args.api_key or os.getenv("HYDRA_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("HYDRA_API_KEY or OPENAI_API_KEY is not set")
    base_url = (
        args.base_url
        or os.getenv("HYDRA_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.hydraai.ru/v1"
    ).rstrip("/")

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    raw_cases = dataset.get("cases")
    if not isinstance(raw_cases, list):
        raise SystemExit("Dataset must contain a cases list")
    if args.case_id:
        requested = set(args.case_id)
        cases = [case for case in raw_cases if case.get("id") in requested]
        missing = requested - {case.get("id") for case in cases}
        if missing:
            raise SystemExit(f"Case IDs not found: {', '.join(sorted(missing))}")
    else:
        cases = raw_cases[: args.limit] if args.limit is not None else raw_cases
    retriever = build_retriever(
        args.themes,
        args.index_dir,
        bm25_weight=args.bm25_weight,
        vector_weight=args.vector_weight,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for model in args.models:
        llm = ChatCompletionCallable(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
        )
        preprocessor = LLMTextPreprocessor(llm=llm)
        for index, case in enumerate(cases, start=1):
            print(f"[{model}] {index}/{len(cases)} {case.get('id')}", flush=True)
            try:
                text = case.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("case text is empty")
                prepared = preprocessor.prepare_parts(text)
                if llm.last_error:
                    raise RuntimeError(llm.last_error)
                candidates, per_query_candidates = search_and_merge(
                    retriever=retriever,
                    retrieval_queries=prepared.retrieval_queries,
                    query_top_k=args.query_top_k,
                    final_top_k=args.final_top_k,
                )
                record = make_record(
                    case=case,
                    model=model,
                    decision_text=prepared.decision_text,
                    decomposition=getattr(preprocessor, "last_retrieval_decomposition", {}) or {},
                    retrieval_queries=prepared.retrieval_queries,
                    candidates=candidates,
                    per_query_candidates=per_query_candidates,
                    usage=llm.last_usage,
                )
            except Exception as error:
                record = make_error_record(case, model, error)
            records.append(record)

    report = {
        "dataset": str(args.dataset),
        "models": args.models,
        "queryTopK": args.query_top_k,
        "finalTopK": args.final_top_k,
        "bm25Weight": args.bm25_weight,
        "vectorWeight": args.vector_weight,
        "usage": sum_usage(records),
        "metrics": calculate_report(records, args.models),
    }
    (args.out_dir / "comparison_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(args.out_dir / "generated_queries.jsonl", records)
    write_csv(args.out_dir / "retrieval_diagnostics.csv", records)

    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Saved report: {args.out_dir / 'comparison_report.json'}")
    print(f"Saved queries: {args.out_dir / 'generated_queries.jsonl'}")
    print(f"Saved diagnostics: {args.out_dir / 'retrieval_diagnostics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
