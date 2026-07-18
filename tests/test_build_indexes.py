import json
from pathlib import Path

from src.retrieval import build_theme_indexes as module


class FakeVector:
    def __init__(self, themes, embedding_model):
        self.themes = themes
        self.embedding_model = embedding_model

    def build(self):
        return None

    def save(self, index_path, embeddings_path, metadata_path):
        index_path.write_bytes(b"index")
        embeddings_path.write_bytes(b"embeddings")
        metadata_path.write_text(
            json.dumps(
                {
                    "codes": [theme.code for theme in self.themes],
                    "embeddingModel": self.embedding_model,
                    "dimension": 3,
                }
            ),
            encoding="utf-8",
        )


def test_build_indexes_writes_complete_artifact_set(monkeypatch):
    root = Path("data/evaluation/test-output")
    themes_path = root / "themes.json"
    generated = [
        root / "bm25.pkl",
        root / "faiss.index",
        root / "embeddings.npy",
        root / "metadata.json",
        root / "index_report.json",
    ]
    themes_path.write_text(
        json.dumps(
            [
                {
                    "code": "0001.0001.0001.0001",
                    "name": "Тема",
                    "section": "0001",
                    "isLeaf": True,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "FaissThemeRetriever", FakeVector)
    try:
        report = module.build_indexes(themes_path, root, "fake-model")
        assert report["themesCount"] == 1
        assert len(report["themesSha256"]) == 64
        assert all(path.exists() for path in generated)
        assert not list(root.glob(".*.tmp"))
    finally:
        for path in [themes_path, *generated]:
            if path.exists():
                path.unlink()
