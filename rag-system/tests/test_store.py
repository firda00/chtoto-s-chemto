import numpy as np
import pytest

from rag_system.schemas import Chunk
from rag_system.store import InMemoryStore, VectorStore, to_pgvector


def _chunks(n: int, collection: str = "c") -> list[Chunk]:
    return [
        Chunk(id=f"{collection}{i}", doc_id="d", text=f"t{i}", collection=collection)
        for i in range(n)
    ]


def test_inmemory_topk_order() -> None:
    store = InMemoryStore()
    vecs = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    store.add(_chunks(4), vecs)
    assert isinstance(store, VectorStore)
    res = store.search(np.array([1, 0, 0]), k=3)
    assert [r.chunk.id for r in res] == ["c0", "c1", "c2"]
    assert res[0].score == pytest.approx(1.0)
    assert res[0].score >= res[1].score >= res[2].score


def test_inmemory_collection_filter_and_upsert() -> None:
    store = InMemoryStore()
    store.add(_chunks(2, "a"), np.array([[1, 0], [0, 1]], dtype=np.float32))
    store.add(_chunks(2, "b"), np.array([[1, 0], [0, 1]], dtype=np.float32))
    assert store.collections() == ["a", "b"]
    res = store.search(np.array([1, 0]), k=5, collection="b")
    assert {r.chunk.collection for r in res} == {"b"}
    assert len(res) == 2
    # upsert по id не дублирует запись
    store.add(
        [Chunk(id="a0", doc_id="d", text="new", collection="a")],
        np.array([[0, 1]], dtype=np.float32),
    )
    assert store.count() == 4
    assert store.all_chunks("a")[0].text == "new"


def test_inmemory_save_load(tmp_path) -> None:
    store = InMemoryStore()
    store.add(_chunks(3), np.eye(3, dtype=np.float32))
    store.save(tmp_path, {"embedder": "hashing"})
    loaded, meta = InMemoryStore.load(tmp_path)
    assert meta["embedder"] == "hashing"
    assert meta["count"] == 3
    assert loaded.search(np.array([0, 0, 1]), k=1)[0].chunk.id == "c2"


def test_empty_store_and_shape_check() -> None:
    store = InMemoryStore()
    assert store.search(np.array([1.0, 0.0]), k=3) == []
    with pytest.raises(ValueError):
        store.add(_chunks(2), np.ones((3, 2)))


def test_pgvector_literal() -> None:
    assert to_pgvector(np.array([0.5, -1.0])) == "[0.5,-1]"
