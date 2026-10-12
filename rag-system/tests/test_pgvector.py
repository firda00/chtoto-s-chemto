"""Интеграционный тест PgVectorStore. Запускается только при заданном PG_DSN, например:

    docker compose up -d
    PG_DSN=postgresql://rag:rag@localhost:5432/rag uv run pytest tests/test_pgvector.py
"""

import os
import uuid

import numpy as np
import pytest

from rag_system.embeddings import HashingEmbedder
from rag_system.schemas import Chunk
from rag_system.store import PgVectorStore, VectorStore

PG_DSN = os.environ.get("PG_DSN")
pytestmark = pytest.mark.skipif(not PG_DSN, reason="PG_DSN не задан — pgvector-тест пропущен")


@pytest.fixture
def pg_store():
    table = f"test_chunks_{uuid.uuid4().hex[:8]}"
    store = PgVectorStore(PG_DSN, dim=64, table=table)
    store.init()
    yield store
    with store._connect() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(f"DROP TABLE IF EXISTS {table}_meta")


def test_pgvector_roundtrip(pg_store: PgVectorStore) -> None:
    assert isinstance(pg_store, VectorStore)
    emb = HashingEmbedder(dim=64)
    texts = {
        "p1": ("pricing", "Гостевой день стоит 120 000 сум"),
        "f1": ("faq", "Color printing costs 3 000 UZS per page"),
        "s1": ("policies", "Dogs under 10 kg are welcome on Fridays"),
    }
    chunks = [Chunk(id=i, doc_id=i, collection=c, text=t, metadata={"source": f"{i}.md"})
              for i, (c, t) in texts.items()]
    pg_store.add(chunks, emb.embed([c.text for c in chunks]))
    pg_store.add(chunks[:1], emb.embed([chunks[0].text]))  # upsert
    assert pg_store.count() == 3
    assert pg_store.collections() == ["faq", "policies", "pricing"]

    res = pg_store.search(emb.embed_query("color printing price"), k=2)
    assert res[0].chunk.id == "f1"
    assert res[0].chunk.metadata["source"] == "f1.md"
    filtered = pg_store.search(emb.embed_query("color printing"), k=5, collection="pricing")
    assert [r.chunk.id for r in filtered] == ["p1"]
    assert len(pg_store.all_chunks("faq")) == 1

    pg_store.set_meta("embedder", "hashing")
    assert pg_store.get_meta()["embedder"] == "hashing"
    pg_store.clear()
    assert pg_store.count() == 0
    assert np.isfinite(emb.embed_query("x")).all()
