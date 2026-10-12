"""End-to-end: data/docs → HashingEmbedder → InMemoryStore → MockLLM."""

from datetime import datetime

import pytest

from rag_system.factory import evaluate, load_eval
from rag_system.llm import MockLLM
from rag_system.modular import ModularRAG
from rag_system.naive import NaiveRAG
from rag_system.rerank import LexicalReranker
from rag_system.router import KeywordRouter

from .conftest import EVAL_PATH, make_runtime


@pytest.mark.parametrize(
    ("question", "expected", "source"),
    [
        ("Сколько стоит гостевой день?", "120 000", "pricing.md"),
        ("How much does color printing cost?", "3 000", "faq.md"),
        ("Are dogs allowed in the coworking?", "10 kg", "policies.md"),
    ],
)
def test_naive_end_to_end(runtime, question: str, expected: str, source: str) -> None:
    rag = runtime.pipeline("naive", k=4)
    assert isinstance(rag, NaiveRAG)
    res = rag.answer(question)
    assert res.mode == "naive"
    assert expected in res.answer
    assert source in [r.chunk.source for r in res.retrieved]
    assert res.citations and res.citations[0].source == source
    # промпт содержит пронумерованный контекст
    prompt = runtime.llm.calls[-1]["prompt"]
    assert "<context>" in prompt and "[1]" in prompt


def test_modular_end_to_end_with_trace(runtime) -> None:
    rag = runtime.pipeline("modular", k=4)
    assert isinstance(rag, ModularRAG)
    res = rag.answer("  Сколько стоит переговорная Луна в час?  ")
    assert "150 000" in res.answer
    assert res.rewritten_query == "Сколько стоит переговорная Луна в час?"
    assert res.route is not None and res.route["collection"] == "pricing"
    assert res.citations[0].source == "pricing.md"
    assert res.retrieved[0].retriever == "rerank"


def test_modular_with_scripted_llm_tool_calls(tmp_path) -> None:
    llm = MockLLM(
        [
            "стоимость переговорной Луна",  # rewrite
            '{"collection": "pricing", "tools": ["calculator"], "reason": "price + math"}',
            {
                "tool_calls": [
                    {"name": "search_docs", "arguments": {"query": "студенческая скидка"}}
                ]
            },
            {"tool_calls": [{"name": "calculator", "arguments": {"expression": "150000 * 3"}}]},
            "Три часа в «Луне» стоят 450 000 сум [1]; скидка студентам 20% [5].",
        ]
    )
    rt = make_runtime(tmp_path, llm=llm)
    rag = ModularRAG(
        rt.store,
        rt.embedder,
        llm,
        reranker=LexicalReranker(),
        k=4,
        now=lambda: datetime(2026, 10, 12),
    )
    res = rag.answer("Сколько стоят 3 часа переговорной Луна?")

    assert res.rewritten_query == "стоимость переговорной Луна"
    assert res.route["router"] == "llm"
    assert [t.name for t in res.tool_calls] == ["search_docs", "calculator"]
    assert res.tool_calls[1].result == "450000"
    assert res.tool_calls[0].result.startswith("[5]")  # нумерация продолжает контекст
    assert {c.n for c in res.citations} == {1, 5}
    # search_docs всегда доступен, calculator — потому что его выбрал роутер, get_date — нет
    tool_names = {t["function"]["name"] for t in llm.calls[2]["tools"]}
    assert tool_names == {"search_docs", "calculator"}


def test_modular_hybrid_beats_bad_route(runtime) -> None:
    """Мягкий роутинг: даже при неверной коллекции правильный фрагмент находится."""
    rag = ModularRAG(
        runtime.store,
        runtime.embedder,
        MockLLM(),
        router=KeywordRouter(),
        k=4,
        use_rewrite=False,
    )
    results = rag.retrieve("Color printing cost per page", collection="schedule")
    assert "faq.md" in [r.chunk.source for r in results]


def test_eval_on_dataset(runtime) -> None:
    items = load_eval(EVAL_PATH)
    assert len(items) >= 10
    for mode in ("naive", "modular"):
        report = evaluate(runtime.pipeline(mode, k=4), items, k=4)
        assert report["n"] == len(items)
        assert report["hit_rate"] >= 0.8
        assert report["answer_accuracy"] >= 0.7


def test_empty_index_message(tmp_path) -> None:
    from rag_system.config import Settings
    from rag_system.factory import Runtime

    rt = Runtime.open(
        Settings(
            llm_provider="mock",
            embedder="hashing",
            reranker="lexical",
            index_dir=str(tmp_path / "none"),
        )
    )
    assert "Индекс пуст" in rt.pipeline("naive").answer("anything").answer
