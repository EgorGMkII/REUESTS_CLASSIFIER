from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from llm_module import OPENAI_EMBEDDING_MODEL

from .bm25_retriever import BM25ThemeRetriever
from .faiss_retriever import FaissThemeRetriever
from .hybrid_theme_retriever import load_themes


def build_indexes(
    themes_path: Path, out_dir: Path, embedding_model: str
) -> dict:
    themes = load_themes(themes_path)
    if not themes:
        raise ValueError("Theme source is empty")
    if any(not theme.isLeaf for theme in themes):
        raise ValueError("Theme source contains non-leaf records")
    codes = [theme.code for theme in themes]
    if len(codes) != len(set(codes)):
        raise ValueError("Theme source contains duplicate codes")

    out_dir.mkdir(parents=True, exist_ok=True)
    bm25_path = out_dir / "bm25.pkl"
    faiss_path = out_dir / "faiss.index"
    embeddings_path = out_dir / "embeddings.npy"
    metadata_path = out_dir / "metadata.json"
    bm25 = BM25ThemeRetriever(themes)
    vector = FaissThemeRetriever(themes, embedding_model)
    vector.build()
    temporary = {
        bm25_path: out_dir / ".bm25.pkl.tmp",
        faiss_path: out_dir / ".faiss.index.tmp",
        embeddings_path: out_dir / ".embeddings.npy.tmp",
        metadata_path: out_dir / ".metadata.json.tmp",
    }
    bm25.save(temporary[bm25_path])
    # np.save adds .npy unless the temporary path already ends in .npy.
    temporary[embeddings_path] = out_dir / ".embeddings.tmp.npy"
    vector.save(
        temporary[faiss_path],
        temporary[embeddings_path],
        temporary[metadata_path],
    )
    for destination, source in temporary.items():
        source.replace(destination)

    report = {
        "themesCount": len(themes),
        "embeddingModel": embedding_model,
        "bm25IndexPath": str(bm25_path),
        "faissIndexPath": str(faiss_path),
        "embeddingsPath": str(embeddings_path),
        "metadataPath": str(metadata_path),
        "themesSha256": hashlib.sha256(themes_path.read_bytes()).hexdigest(),
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    (out_dir / "index_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build local theme indexes")
    parser.add_argument("--themes", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--embedding-model", default=OPENAI_EMBEDDING_MODEL
    )
    parser.add_argument(
        "--use-openai-embeddings",
        action="store_true",
        help="Compatibility flag; OpenAI-compatible embeddings are the default",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = build_indexes(args.themes, args.out_dir, args.embedding_model)
    print("Theme indexes built.")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
