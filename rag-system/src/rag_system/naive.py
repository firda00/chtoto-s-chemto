"""Naive RAG: embed(query) → cosine top-k → контекст в промпт → ответ LLM.

    question ──► Embedder ──► VectorStore.search(top_k) ──► prompt(context) ──► LLM ──► answer
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rag_system.embeddings import Embedder
from rag_system.llm import LLM
from rag_system.prompts import ANSWER_SYSTEM, build_answer_prompt
from rag_system.schemas import Citation, RAGAnswer, SearchResult
from rag_system.store import VectorStore

_CITE = re.compile(r"\[(\d+)\]")


def make_citations(answer: str, results: Sequence[SearchResult]) -> list[Citation]:
    """Цитаты — фрагменты, на которые ссылается ответ ([n]); если ссылок нет — все фрагменты."""
    referenced = {int(n) for n in _CITE.findall(answer)}
    out = []
    for n, r in enumerate(results, start=1):
        if referenced and n not in referenced:
            continue
        snippet = " ".join(r.chunk.text.split())
        out.append(
            Citation(
                n=n,
                chunk_id=r.chunk.id,
                source=r.chunk.source,
                heading=r.chunk.heading,
                snippet=snippet[:160] + ("…" if len(snippet) > 160 else ""),
            )
        )
    return out


class NaiveRAG:
    mode = "naive"

    def __init__(self, store: VectorStore, embedder: Embedder, llm: LLM, k: int = 4) -> None:
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.k = k

    def retrieve(self, question: str, k: int | None = None) -> list[SearchResult]:
        return self.store.search(self.embedder.embed_query(question), k=k or self.k)

    def answer(self, question: str) -> RAGAnswer:
        results = self.retrieve(question)
        if not results:
            return RAGAnswer(
                question=question,
                answer="Индекс пуст — сначала выполните `rag ingest <path>`.",
                mode=self.mode,
            )
        raw = self.llm.complete(build_answer_prompt(question, results), system=ANSWER_SYSTEM)
        text = raw if isinstance(raw, str) else str(raw)
        return RAGAnswer(
            question=question,
            answer=text,
            mode=self.mode,
            citations=make_citations(text, results),
            retrieved=results,
        )
