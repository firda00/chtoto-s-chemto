"""Векторные хранилища за единым протоколом VectorStore.

* InMemoryStore — numpy-матрица + косинусное сходство; умеет сохраняться в папку
  (vectors.npy + chunks.jsonl + meta.json), чтобы `rag ingest` и `rag ask` работали
  в разных процессах без внешних сервисов.
* PgVectorStore — Postgres + pgvector через psycopg3. `init()` — миграция:
  расширение vector, таблица чанков, HNSW/IVFFlat-индекс по косинусному расстоянию.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from rag_system.embeddings import l2_normalize
from rag_system.schemas import Chunk, SearchResult


@runtime_checkable
class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None: ...

    def search(
        self, query_embedding: np.ndarray, k: int = 4, collection: str | None = None
    ) -> list[SearchResult]: ...

    def all_chunks(self, collection: str | None = None) -> list[Chunk]: ...

    def collections(self) -> list[str]: ...

    def count(self) -> int: ...

    def clear(self) -> None: ...


def _check_shapes(chunks: Sequence[Chunk], embeddings: np.ndarray) -> np.ndarray:
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[0] != len(chunks):
        raise ValueError(f"embeddings shape {emb.shape} не соответствует {len(chunks)} чанкам")
    return emb


class InMemoryStore:
    """Простое хранилище на numpy. Поиск — полный перебор (для портфолио-объёмов достаточно)."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim
        self._chunks: list[Chunk] = []
        self._index: dict[str, int] = {}
        self._matrix = np.zeros((0, dim or 0), dtype=np.float32)

    # --- запись -------------------------------------------------------------------------------
    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        emb = l2_normalize(_check_shapes(chunks, embeddings))
        if self.dim is None or self._matrix.shape[0] == 0:
            self.dim = emb.shape[1]
            if self._matrix.shape[0] == 0:
                self._matrix = np.zeros((0, self.dim), dtype=np.float32)
        if emb.shape[1] != self.dim:
            raise ValueError(f"Размерность {emb.shape[1]} != размерности хранилища {self.dim}")
        new_rows: list[np.ndarray] = []
        for chunk, vec in zip(chunks, emb, strict=True):
            if chunk.id in self._index:  # upsert
                i = self._index[chunk.id]
                self._chunks[i] = chunk
                self._matrix[i] = vec
            else:
                self._index[chunk.id] = len(self._chunks) + len(new_rows)
                new_rows.append(vec)
                self._chunks.append(chunk)
        if new_rows:
            self._matrix = np.vstack([self._matrix, np.stack(new_rows)])

    def clear(self) -> None:
        self._chunks.clear()
        self._index.clear()
        self._matrix = np.zeros((0, self.dim or 0), dtype=np.float32)

    # --- чтение -------------------------------------------------------------------------------
    def search(
        self, query_embedding: np.ndarray, k: int = 4, collection: str | None = None
    ) -> list[SearchResult]:
        if not self._chunks or k <= 0:
            return []
        q = l2_normalize(np.asarray(query_embedding, dtype=np.float32).reshape(-1))
        sims = self._matrix @ q  # строки уже нормированы → это косинус
        if collection:
            mask = np.array([c.collection == collection for c in self._chunks])
            sims = np.where(mask, sims, -np.inf)
        k = min(k, len(self._chunks))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top], kind="stable")]
        return [
            SearchResult(chunk=self._chunks[i], score=float(sims[i]), retriever="vector")
            for i in top
            if np.isfinite(sims[i])
        ]

    def all_chunks(self, collection: str | None = None) -> list[Chunk]:
        return [c for c in self._chunks if collection is None or c.collection == collection]

    def collections(self) -> list[str]:
        return sorted({c.collection for c in self._chunks})

    def count(self) -> int:
        return len(self._chunks)

    # --- персистентность ----------------------------------------------------------------------
    def save(self, directory: str | Path, meta: dict[str, Any] | None = None) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self._matrix)
        with (d / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for c in self._chunks:
                fh.write(c.model_dump_json() + "\n")
        info = {**(meta or {}), "dim": self.dim, "count": self.count()}
        (d / "meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), "utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> tuple[InMemoryStore, dict[str, Any]]:
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text("utf-8"))
        store = cls(dim=meta.get("dim"))
        matrix = np.load(d / "vectors.npy")
        with (d / "chunks.jsonl").open(encoding="utf-8") as fh:
            chunks = [Chunk.model_validate_json(line) for line in fh if line.strip()]
        if chunks:
            store.add(chunks, matrix)
        return store, meta

    @staticmethod
    def exists(directory: str | Path) -> bool:
        return (Path(directory) / "meta.json").exists()


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


def to_pgvector(vec: np.ndarray) -> str:
    """Текстовый литерал pgvector: '[0.1,0.2,...]' (без зависимости от pgvector-python)."""
    return "[" + ",".join(f"{float(x):.7g}" for x in np.asarray(vec).reshape(-1)) + "]"


class PgVectorStore:
    """Postgres + pgvector. Сходство — 1 - cosine_distance (оператор `<=>`)."""

    def __init__(
        self, dsn: str, dim: int, table: str = "rag_chunks", index_type: str = "hnsw"
    ) -> None:
        if not _IDENT.match(table):
            raise ValueError(f"Недопустимое имя таблицы: {table!r}")
        if index_type not in {"hnsw", "ivfflat"}:
            raise ValueError("index_type должен быть hnsw или ivfflat")
        self.dsn = dsn
        self.dim = dim
        self.table = table
        self.index_type = index_type

    def _connect(self) -> Any:
        import psycopg  # ленивый импорт: libpq нужен только при реальной работе с БД

        return psycopg.connect(self.dsn, autocommit=True)

    # --- миграция -----------------------------------------------------------------------------
    def init(self) -> None:
        """Создаёт расширение, таблицы и индексы (идемпотентно)."""
        t = self.table
        index_sql = (
            f"CREATE INDEX IF NOT EXISTS {t}_embedding_hnsw ON {t} "
            "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
            if self.index_type == "hnsw"
            else f"CREATE INDEX IF NOT EXISTS {t}_embedding_ivf ON {t} "
            "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
        )
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {t} (
                    id          TEXT PRIMARY KEY,
                    doc_id      TEXT NOT NULL,
                    collection  TEXT NOT NULL DEFAULT 'default',
                    text        TEXT NOT NULL,
                    metadata    JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding   vector({int(self.dim)}) NOT NULL
                )"""
            )
            conn.execute(f"CREATE INDEX IF NOT EXISTS {t}_collection ON {t} (collection)")
            conn.execute(index_sql)
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {t}_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

    def set_meta(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO {self.table}_meta (key, value) VALUES (%s, %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                (key, value),
            )

    def get_meta(self) -> dict[str, str]:
        with self._connect() as conn:
            try:
                rows = conn.execute(f"SELECT key, value FROM {self.table}_meta").fetchall()
            except Exception:  # таблицы ещё нет
                return {}
        return {k: v for k, v in rows}

    # --- запись -------------------------------------------------------------------------------
    def add(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        emb = l2_normalize(_check_shapes(chunks, embeddings))
        rows = [
            (c.id, c.doc_id, c.collection, c.text, json.dumps(c.metadata, ensure_ascii=False),
             to_pgvector(v))
            for c, v in zip(chunks, emb, strict=True)
        ]
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                f"""INSERT INTO {self.table} (id, doc_id, collection, text, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET
                        doc_id = EXCLUDED.doc_id, collection = EXCLUDED.collection,
                        text = EXCLUDED.text, metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding""",
                rows,
            )

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute(f"TRUNCATE {self.table}")

    # --- чтение -------------------------------------------------------------------------------
    @staticmethod
    def _row_to_chunk(row: Sequence[Any]) -> Chunk:
        cid, doc_id, collection, text, metadata = row[:5]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return Chunk(id=cid, doc_id=doc_id, collection=collection, text=text, metadata=metadata)

    def search(
        self, query_embedding: np.ndarray, k: int = 4, collection: str | None = None
    ) -> list[SearchResult]:
        q = to_pgvector(l2_normalize(np.asarray(query_embedding).reshape(-1)))
        where = "WHERE collection = %s" if collection else ""
        params: list[Any] = [q]
        if collection:
            params.append(collection)
        params += [q, int(k)]
        sql = f"""SELECT id, doc_id, collection, text, metadata,
                         1 - (embedding <=> %s::vector) AS score
                  FROM {self.table} {where}
                  ORDER BY embedding <=> %s::vector
                  LIMIT %s"""
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            SearchResult(chunk=self._row_to_chunk(r), score=float(r[5]), retriever="vector")
            for r in rows
        ]

    def all_chunks(self, collection: str | None = None) -> list[Chunk]:
        sql = f"SELECT id, doc_id, collection, text, metadata FROM {self.table}"
        params: list[Any] = []
        if collection:
            sql += " WHERE collection = %s"
            params.append(collection)
        sql += " ORDER BY doc_id, id"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def collections(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT collection FROM {self.table} ORDER BY 1"
            ).fetchall()
        return [r[0] for r in rows]

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0])
