from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .bm25_retriever import BM25ThemeRetriever
from .faiss_retriever import FaissThemeRetriever
from .text_normalization import normalize_text
from .theme_schema import ThemeCandidate, ThemeRecord, candidate_from_theme


def _dump(model: Any) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def load_themes(path: Path) -> list[ThemeRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [ThemeRecord(**item) for item in raw]


class HybridThemeRetriever:
    def __init__(
        self,
        bm25: BM25ThemeRetriever,
        vector: FaissThemeRetriever,
        bm25_weight: float = 0.45,
        vector_weight: float = 0.55,
    ):
        if bm25_weight < 0 or vector_weight < 0 or bm25_weight + vector_weight <= 0:
            raise ValueError("Retriever weights must be non-negative and non-zero")
        self.bm25 = bm25
        self.vector = vector
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

    def search(
        self,
        query: str,
        top_k: int = 50,
        bm25_k: int = 80,
        vector_k: int = 80,
        section: str | None = None,
    ) -> list[ThemeCandidate]:
        if not normalize_text(query) or top_k <= 0:
            return []
        lexical = self.bm25.search(query, bm25_k, section)
        semantic = self.vector.search(query, vector_k, section)
        merged: dict[str, dict[str, Any]] = {}
        for candidate in lexical:
            merged[candidate.code] = {
                "theme": candidate,
                "bm25": candidate.bm25Score,
                "vector": None,
            }
        for candidate in semantic:
            entry = merged.setdefault(
                candidate.code,
                {"theme": candidate, "bm25": None, "vector": None},
            )
            entry["vector"] = candidate.vectorScore

        scored = []
        for code, entry in merged.items():
            bm25_score = entry["bm25"]
            vector_score = entry["vector"]
            score = self.bm25_weight * (bm25_score or 0.0)
            score += self.vector_weight * (vector_score or 0.0)
            if bm25_score is not None and vector_score is not None:
                score += 0.05
            scored.append((min(1.0, max(0.0, score)), code, entry))
        scored.sort(key=lambda item: (-item[0], item[1]))

        results = []
        theme_fields = set(
            ThemeRecord.model_fields
            if hasattr(ThemeRecord, "model_fields")
            else ThemeRecord.__fields__
        )
        for rank, (score, _, entry) in enumerate(scored[:top_k], 1):
            source = entry["theme"]
            theme = ThemeRecord(
                **{
                    key: value
                    for key, value in _dump(source).items()
                    if key in theme_fields
                }
            )
            results.append(
                candidate_from_theme(
                    theme,
                    bm25Score=entry["bm25"],
                    vectorScore=entry["vector"],
                    hybridScore=score,
                    rank=rank,
                    source="hybrid",
                )
            )
        return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search thematic classifier")
    parser.add_argument("--themes", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--section")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    themes = load_themes(args.themes)
    metadata = json.loads(
        (args.index_dir / "metadata.json").read_text(encoding="utf-8")
    )
    bm25 = BM25ThemeRetriever.load(args.index_dir / "bm25.pkl", themes)
    vector = FaissThemeRetriever.load(
        themes,
        args.index_dir / "faiss.index",
        args.index_dir / "embeddings.npy",
        args.index_dir / "metadata.json",
        metadata["embeddingModel"],
    )
    results = HybridThemeRetriever(bm25, vector).search(
        args.query, args.top_k, section=args.section
    )
    if args.json:
        print(json.dumps([_dump(item) for item in results], ensure_ascii=False, indent=2))
        return 0
    print(f"Query: {args.query}\n")
    for item in results:
        bm25_score = f"{item.bm25Score:.2f}" if item.bm25Score is not None else "-"
        vector_score = (
            f"{item.vectorScore:.2f}" if item.vectorScore is not None else "-"
        )
        print(f"{item.rank}. {item.code} | {item.name}")
        print(
            f"   hybrid={item.hybridScore:.2f} "
            f"bm25={bm25_score} vector={vector_score}"
        )
        print(f"   path={' > '.join(item.path)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
