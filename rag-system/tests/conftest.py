from __future__ import annotations

from pathlib import Path

import pytest

from rag_system.config import Settings
from rag_system.embeddings import HashingEmbedder
from rag_system.factory import Runtime
from rag_system.llm import MockLLM
from rag_system.rerank import LexicalReranker

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "docs"
EVAL_PATH = ROOT / "data" / "eval.jsonl"


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Тесты никогда не ходят в сеть: принудительно mock LLM и hashing-эмбеддер."""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDER", "hashing")
    monkeypatch.setenv("RERANKER", "lexical")
    monkeypatch.setenv("RAG_STORE", "memory")
    monkeypatch.delenv("LLM_API_KEY", raising=False)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


def make_runtime(tmp_path: Path, llm: MockLLM | None = None) -> Runtime:
    settings = Settings(
        llm_provider="mock",
        embedder="hashing",
        reranker="lexical",
        store="memory",
        index_dir=str(tmp_path / "index"),
    )
    rt = Runtime.open(settings, llm=llm or MockLLM(), reranker=LexicalReranker(), fresh=True)
    rt.ingest(DOCS_DIR)
    return rt


@pytest.fixture
def runtime(tmp_path: Path) -> Runtime:
    return make_runtime(tmp_path)
