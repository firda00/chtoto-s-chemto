import pytest

from rag_system.bm25 import BM25Index
from rag_system.fusion import reciprocal_rank_fusion
from rag_system.rerank import LexicalReranker
from rag_system.schemas import Chunk, SearchResult


def _c(i: str, text: str = "") -> Chunk:
    return Chunk(id=i, doc_id=i, text=text or i)


def _r(*ids: str) -> list[SearchResult]:
    return [SearchResult(chunk=_c(i), score=1.0) for i in ids]


def test_bm25_ranks_lexical_match_first() -> None:
    chunks = [
        _c("pets", "Dogs under 10 kg are welcome on Fridays"),
        _c("price", "Гостевой день стоит 120 000 сум"),
        _c("print", "Color printing costs 3 000 UZS per page"),
    ]
    idx = BM25Index(chunks)
    assert idx.search("сколько стоит гостевой день", k=3)[0].chunk.id == "price"
    assert idx.search("are dogs allowed", k=3)[0].chunk.id == "pets"
    assert idx.search("квантовая физика", k=3) == []
    assert BM25Index([]).search("anything") == []


def test_rrf_orders_by_combined_rank() -> None:
    vector = _r("a", "b", "c")
    bm25 = _r("b", "c", "d")
    fused = reciprocal_rank_fusion([vector, bm25], k=60)
    # b: 1/62+1/61, c: 1/63+1/62, a: 1/61, d: 1/63
    assert [r.chunk.id for r in fused] == ["b", "c", "a", "d"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert all(r.retriever == "rrf" for r in fused)


def test_rrf_weights_and_top_n() -> None:
    fused = reciprocal_rank_fusion([_r("a", "b"), _r("b", "a")], weights=[2.0, 1.0], top_n=1)
    assert [r.chunk.id for r in fused] == ["a"]


def test_rrf_ignores_duplicates_within_list() -> None:
    fused = reciprocal_rank_fusion([_r("a", "a", "b")])
    assert [r.chunk.id for r in fused] == ["a", "b"]


def test_lexical_reranker_promotes_best_match() -> None:
    results = [
        SearchResult(chunk=_c("x", "Open area works from 9:00 to 21:00"), score=1.0),
        SearchResult(chunk=_c("y", "Color printing costs 3 000 UZS per page"), score=0.5),
    ]
    out = LexicalReranker().rerank("color printing cost", results, top_k=1)
    assert [r.chunk.id for r in out] == ["y"]
    assert out[0].retriever == "rerank"
