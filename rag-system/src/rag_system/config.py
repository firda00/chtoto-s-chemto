"""Настройки из переменных окружения (+ необязательный .env в текущей папке)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from rag_system.embeddings import DEFAULT_FASTEMBED_MODEL
from rag_system.rerank import DEFAULT_RERANK_MODEL


def load_dotenv(path: str | Path = ".env") -> None:
    """Минимальный загрузчик .env: KEY=VALUE, не перетирает уже заданные переменные."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        if key and value and key not in os.environ:
            os.environ[key] = value


@dataclass
class Settings:
    llm_provider: str = "auto"
    embedder: str = "auto"
    embed_model: str = DEFAULT_FASTEMBED_MODEL
    reranker: str = "auto"
    rerank_model: str = DEFAULT_RERANK_MODEL
    store: str = "memory"
    index_dir: str = ".rag_index"
    pg_dsn: str = "postgresql://rag:rag@localhost:5432/rag"
    pg_index: str = "hnsw"
    pg_table: str = "rag_chunks"
    top_k: int = 4

    @classmethod
    def from_env(cls) -> Settings:
        e = os.environ.get
        return cls(
            llm_provider=e("LLM_PROVIDER", "auto"),
            embedder=e("EMBEDDER", "auto"),
            embed_model=e("EMBED_MODEL", DEFAULT_FASTEMBED_MODEL),
            reranker=e("RERANKER", "auto"),
            rerank_model=e("RERANK_MODEL", DEFAULT_RERANK_MODEL),
            store=e("RAG_STORE", "memory"),
            index_dir=e("RAG_INDEX_DIR", ".rag_index"),
            pg_dsn=e("PG_DSN", "postgresql://rag:rag@localhost:5432/rag"),
            pg_index=e("PG_INDEX", "hnsw"),
            pg_table=e("PG_TABLE", "rag_chunks"),
            top_k=int(e("RAG_TOP_K", "4")),
        )
