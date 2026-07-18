from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from .text_normalization import make_theme_search_document, normalize_text
from .theme_schema import ThemeCandidate, ThemeRecord, candidate_from_theme

EmbeddingFunction = Callable[[str], list[float]]
BatchEmbeddingFunction = Callable[[list[str]], list[list[float]]]


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("Embeddings must be a two-dimensional array")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding provider returned a zero vector")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


class FaissThemeRetriever:
    def __init__(
        self,
        themes: list[ThemeRecord],
        embedding_model_name: str = "text-embedding-3-small",
        embedding_fn: EmbeddingFunction | None = None,
        embedding_batch_fn: BatchEmbeddingFunction | None = None,
    ):
        self.themes = list(themes)
        self.embedding_model_name = embedding_model_name
        self._embedding_fn = embedding_fn
        self._embedding_batch_fn = embedding_batch_fn
        self.embeddings: np.ndarray | None = None
        self.index = None

    def _embed_one(self, text: str) -> list[float]:
        if self._embedding_fn is not None:
            return self._embedding_fn(text)
        from llm_module import get_openai_embedding

        return get_openai_embedding(text)

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        if self._embedding_batch_fn is not None:
            return self._embedding_batch_fn(texts)
        if self._embedding_fn is not None:
            return [self._embedding_fn(text) for text in texts]
        from llm_module import get_openai_embeddings

        return get_openai_embeddings(texts)

    def _make_index(self) -> None:
        if self.embeddings is None:
            raise RuntimeError("Embeddings have not been built")
        try:
            import faiss
        except ImportError:
            self.index = None
            return
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def build(self, batch_size: int = 100) -> None:
        documents = [make_theme_search_document(theme) for theme in self.themes]
        vectors: list[list[float]] = []
        for start in range(0, len(documents), batch_size):
            vectors.extend(self._embed_many(documents[start : start + batch_size]))
        if len(vectors) != len(self.themes):
            raise ValueError("Embedding count does not match theme count")
        self.embeddings = _normalize_vectors(np.asarray(vectors, dtype=np.float32))
        self._make_index()

    def search(
        self, query: str, top_k: int = 50, section: str | None = None
    ) -> list[ThemeCandidate]:
        if not normalize_text(query) or top_k <= 0:
            return []
        if self.embeddings is None:
            raise RuntimeError("Vector index is not built or loaded")
        allowed = np.asarray(
            [
                index
                for index, theme in enumerate(self.themes)
                if section is None or theme.section == section
            ],
            dtype=np.int64,
        )
        if allowed.size == 0:
            return []
        query_vector = _normalize_vectors(np.asarray([self._embed_one(query)]))
        # Searching the small, filtered matrix is deterministic and guarantees
        # enough results even when a section filter is active.
        cosine = self.embeddings[allowed] @ query_vector[0]
        order = sorted(
            range(len(allowed)),
            key=lambda position: (
                -float(cosine[position]),
                self.themes[int(allowed[position])].code,
            ),
        )[:top_k]
        results = []
        for rank, position in enumerate(order, 1):
            index = int(allowed[position])
            score = max(0.0, min(1.0, (float(cosine[position]) + 1.0) / 2.0))
            results.append(
                candidate_from_theme(
                    self.themes[index],
                    vectorScore=score,
                    hybridScore=score,
                    rank=rank,
                    source="vector",
                )
            )
        return results

    def save(
        self, index_path: Path, embeddings_path: Path, metadata_path: Path
    ) -> None:
        if self.embeddings is None:
            raise RuntimeError("Vector index is not built")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError("Saving faiss.index requires faiss-cpu") from error
        if self.index is None:
            self._make_index()
        faiss.write_index(self.index, str(index_path))
        np.save(embeddings_path, self.embeddings)
        metadata_path.write_text(
            json.dumps(
                {
                    "codes": [theme.code for theme in self.themes],
                    "embeddingModel": self.embedding_model_name,
                    "dimension": int(self.embeddings.shape[1]),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        themes: list[ThemeRecord],
        index_path: Path,
        embeddings_path: Path,
        metadata_path: Path,
        embedding_model_name: str,
        embedding_fn: EmbeddingFunction | None = None,
    ) -> "FaissThemeRetriever":
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["codes"] != [theme.code for theme in themes]:
            raise ValueError("FAISS metadata does not match the supplied themes")
        if metadata.get("embeddingModel") != embedding_model_name:
            raise ValueError("Embedding model does not match the saved index")
        instance = cls(themes, embedding_model_name, embedding_fn=embedding_fn)
        instance.embeddings = np.load(embeddings_path).astype(np.float32)
        if instance.embeddings.shape[0] != len(themes):
            raise ValueError("Embedding matrix does not match theme count")
        try:
            import faiss
        except ImportError:
            instance.index = None
        else:
            instance.index = faiss.read_index(str(index_path))
        return instance
