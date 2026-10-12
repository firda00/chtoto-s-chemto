"""Сборка компонентов из настроек: эмбеддер + хранилище + LLM + пайплайны, ingest и eval."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_system.chunking import Chunker, chunk_documents, get_chunker, load_documents
from rag_system.config import Settings
from rag_system.embeddings import Embedder, get_embedder
from rag_system.llm import LLM, get_llm
from rag_system.modular import ModularRAG
from rag_system.naive import NaiveRAG
from rag_system.rerank import Reranker, get_reranker
from rag_system.store import InMemoryStore, PgVectorStore, VectorStore

log = logging.getLogger(__name__)


@dataclass
class Runtime:
    settings: Settings
    embedder: Embedder
    store: VectorStore
    llm: LLM
    reranker: Reranker
    meta: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------------------------------
    @classmethod
    def open(
        cls,
        settings: Settings | None = None,
        *,
        llm: LLM | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        fresh: bool = False,
    ) -> Runtime:
        """fresh=True — игнорировать существующий индекс (для `ingest --reset`)."""
        s = settings or Settings.from_env()
        meta: dict[str, Any] = {}
        store: VectorStore
        if s.store == "memory":
            if not fresh and InMemoryStore.exists(s.index_dir):
                mem, meta = InMemoryStore.load(s.index_dir)
                store = mem
            else:
                store = InMemoryStore()
            embedder = embedder or _embedder_for(s, meta)
        elif s.store == "pgvector":
            embedder = embedder or get_embedder(s.embedder, s.embed_model)
            pg = PgVectorStore(s.pg_dsn, dim=embedder.dim, table=s.pg_table, index_type=s.pg_index)
            meta = {} if fresh else pg.get_meta()
            if s.embedder == "auto" and meta.get("embedder") and meta["embedder"] != embedder.name:
                embedder = _embedder_for(s, meta)
                pg.dim = embedder.dim
            store = pg
        else:
            raise ValueError(f"Неизвестный RAG_STORE: {s.store!r} (memory | pgvector)")
        _check_compatible(embedder, meta)
        return cls(
            settings=s,
            embedder=embedder,
            store=store,
            llm=llm or get_llm(s.llm_provider),
            reranker=reranker or get_reranker(s.reranker, s.rerank_model),
            meta=meta,
        )

    # ---------------------------------------------------------------------------------------
    def pipeline(self, mode: str = "modular", k: int | None = None) -> NaiveRAG | ModularRAG:
        k = k or self.settings.top_k
        if mode == "naive":
            return NaiveRAG(self.store, self.embedder, self.llm, k=k)
        if mode == "modular":
            return ModularRAG(self.store, self.embedder, self.llm, reranker=self.reranker, k=k)
        raise ValueError(f"mode должен быть naive или modular, а не {mode!r}")

    def init_db(self) -> None:
        if not isinstance(self.store, PgVectorStore):
            raise RuntimeError("init-db нужен только для RAG_STORE=pgvector")
        self.store.init()
        self.store.set_meta("embedder", self.embedder.name)
        self.store.set_meta("embed_model", getattr(self.embedder, "model_name", ""))
        self.store.set_meta("dim", str(self.embedder.dim))

    def ingest(
        self, path: str | Path, chunker: Chunker | None = None, reset: bool = False,
        batch_size: int = 64,
    ) -> dict[str, int]:
        chunker = chunker or get_chunker("markdown")
        docs = load_documents(path)
        chunks = chunk_documents(docs, chunker)
        if isinstance(self.store, PgVectorStore):
            self.init_db()
        if reset:
            self.store.clear()
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self.store.add(batch, self.embedder.embed([c.text for c in batch]))
        self.save()
        stats: dict[str, int] = {}
        for c in chunks:
            stats[c.collection] = stats.get(c.collection, 0) + 1
        return {"documents": len(docs), "chunks": len(chunks), **stats}

    def save(self) -> None:
        if isinstance(self.store, InMemoryStore):
            self.meta = {
                "embedder": self.embedder.name,
                "embed_model": getattr(self.embedder, "model_name", ""),
            }
            self.store.save(self.settings.index_dir, self.meta)


def _embedder_for(s: Settings, meta: dict[str, Any]) -> Embedder:
    """При EMBEDDER=auto используем тот эмбеддер, которым индекс был построен."""
    if s.embedder == "auto" and meta.get("embedder"):
        return get_embedder(meta["embedder"], meta.get("embed_model") or s.embed_model)
    return get_embedder(s.embedder, s.embed_model)


def _check_compatible(embedder: Embedder, meta: dict[str, Any]) -> None:
    built = meta.get("embedder")
    if built and built != embedder.name:
        raise RuntimeError(
            f"Индекс построен эмбеддером {built!r}, а сейчас выбран {embedder.name!r}. "
            "Переиндексируйте: `rag ingest <path> --reset` или верните EMBEDDER."
        )


# -------------------------------------------------------------------------------------------
# eval
# -------------------------------------------------------------------------------------------
def load_eval(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate(
    pipeline: NaiveRAG | ModularRAG, items: Sequence[dict[str, str]], k: int
) -> dict[str, Any]:
    """hit-rate@k: среди top-k фрагментов есть ожидаемый источник;
    answer-contains: ожидаемая подстрока есть в ответе (без учёта регистра)."""
    rows = []
    for item in items:
        res = pipeline.answer(item["question"])
        sources = [r.chunk.source for r in res.retrieved[:k]]
        hit = item.get("source", "") in sources
        contains = item["expected"].lower() in res.answer.lower()
        rows.append({"question": item["question"], "hit": hit, "contains": contains,
                     "answer": res.answer, "sources": sources})
    n = max(len(rows), 1)
    return {
        "mode": pipeline.mode,
        "k": k,
        "n": len(rows),
        "hit_rate": sum(r["hit"] for r in rows) / n,
        "answer_accuracy": sum(r["contains"] for r in rows) / n,
        "rows": rows,
    }
