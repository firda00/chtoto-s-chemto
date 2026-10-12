"""Modular RAG: каждый этап — заменяемый модуль.

    question
      │ 1. preprocess (нормализация)          text.normalize_query
      │ 2. rewrite (LLM → поисковый запрос)    prompts.REWRITE_*
      │ 3. route (коллекция + инструменты)     router.LLMRouter / KeywordRouter
      │    (мягкий: бонус выбранной коллекции в RRF, а не жёсткий фильтр)
      ▼
    hybrid retrieval: [vector top-N, BM25 top-N] × {исходный, переписанный запрос}
      │ 4. RRF fusion                          fusion.reciprocal_rank_fusion
      │ 5. rerank → top-k                      rerank.FastEmbedReranker / LexicalReranker
      ▼
    6. generation + tool loop (search_docs / calculator / get_date)  tools.run_tool_loop
      ▼
    answer + citations [n] + trace (rewrite, route, tool calls)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime

from rag_system.bm25 import BM25Index
from rag_system.embeddings import Embedder
from rag_system.fusion import reciprocal_rank_fusion
from rag_system.llm import LLM
from rag_system.naive import make_citations
from rag_system.prompts import (
    ANSWER_SYSTEM,
    REWRITE_SYSTEM,
    build_answer_prompt,
    build_rewrite_prompt,
    format_context,
)
from rag_system.rerank import LexicalReranker, Reranker
from rag_system.router import LLMRouter, Route, Router
from rag_system.schemas import RAGAnswer, SearchResult
from rag_system.store import VectorStore
from rag_system.text import normalize_query
from rag_system.tools import default_registry, run_tool_loop

log = logging.getLogger(__name__)

TOOL_SYSTEM = (
    ANSWER_SYSTEM
    + " You may call tools: search_docs to look up more fragments (cite them by their numbers), "
    "calculator for any arithmetic, get_date for the current date. Call tools only when needed."
)


class ModularRAG:
    mode = "modular"

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        llm: LLM,
        *,
        reranker: Reranker | None = None,
        router: Router | None = None,
        k: int = 4,
        candidates: int = 20,
        rrf_k: int = 60,
        route_weight: float = 0.5,
        use_rewrite: bool = True,
        use_tools: bool = True,
        max_tool_steps: int = 4,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.reranker = reranker or LexicalReranker()
        self.router = router or LLMRouter(llm)
        self.k = k
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.route_weight = route_weight
        self.use_rewrite = use_rewrite
        self.use_tools = use_tools
        self.max_tool_steps = max_tool_steps
        self.now = now
        self._bm25_cache: dict[str | None, tuple[int, BM25Index]] = {}

    # --- 1-2. preprocessing & rewriting ---------------------------------------------------------
    def rewrite(self, query: str) -> str:
        if not self.use_rewrite:
            return query
        try:
            raw = self.llm.complete(build_rewrite_prompt(query), system=REWRITE_SYSTEM)
        except Exception as exc:  # переписывание — оптимизация, не точка отказа
            log.warning("rewrite failed: %s", exc)
            return query
        text = normalize_query(raw if isinstance(raw, str) else "")
        text = text.splitlines()[0] if text else ""
        if not text or len(text) > 3 * len(query) + 50:
            return query
        return text

    # --- 4-5. hybrid retrieval ------------------------------------------------------------------
    def _bm25(self, collection: str | None) -> BM25Index:
        total = self.store.count()
        cached = self._bm25_cache.get(collection)
        if cached is None or cached[0] != total:
            cached = (total, BM25Index(self.store.all_chunks(collection)))
            self._bm25_cache[collection] = cached
        return cached[1]

    def _rankings(self, queries: list[str], collection: str | None) -> list[list[SearchResult]]:
        out: list[list[SearchResult]] = []
        for q in dict.fromkeys(q for q in queries if q):
            vec = self.embedder.embed_query(q)
            out.append(self.store.search(vec, k=self.candidates, collection=collection))
            out.append(self._bm25(collection).search(q, k=self.candidates))
        return out

    def hybrid_candidates(
        self, queries: list[str], collection: str | None = None
    ) -> list[SearchResult]:
        """Vector + BM25 по всем запросам → RRF.

        Мягкий роутинг: если роутер выбрал коллекцию, к глобальным спискам добавляются
        списки только по этой коллекции (вес route_weight). Фрагменты выбранной коллекции
        получают бонус, но ошибка роутера не отрезает правильный ответ полностью.
        """
        rankings = self._rankings(queries, None)
        weights = [1.0] * len(rankings)
        if collection:
            routed = self._rankings(queries, collection)
            rankings += routed
            weights += [self.route_weight] * len(routed)
        return reciprocal_rank_fusion(
            rankings, k=self.rrf_k, weights=weights, top_n=self.candidates
        )

    def retrieve(
        self, question: str, queries: list[str] | None = None, collection: str | None = None,
        k: int | None = None,
    ) -> list[SearchResult]:
        fused = self.hybrid_candidates(queries or [question], collection)
        return self.reranker.rerank(question, fused, top_k=k or self.k)

    # --- 6. generation --------------------------------------------------------------------------
    def answer(self, question: str) -> RAGAnswer:
        q = normalize_query(question)
        rewritten = self.rewrite(q)
        route: Route = self.router.route(rewritten, self.store.collections())
        results = self.retrieve(q, [q, rewritten], route.collection)
        if not results:
            return RAGAnswer(
                question=question,
                answer="Индекс пуст — сначала выполните `rag ingest <path>`.",
                mode=self.mode,
                rewritten_query=rewritten,
                route=route.model_dump(),
            )

        sources: list[SearchResult] = list(results)

        def search_docs(query: str, collection: str | None = None) -> str:
            found = self.retrieve(query, [query], collection)
            known = {s.chunk.id for s in sources}
            new = [r for r in found if r.chunk.id not in known]
            start = len(sources) + 1
            sources.extend(new)
            return format_context(new, start=start) if new else "No new fragments found."

        prompt = build_answer_prompt(q, results)
        if self.use_tools:
            # search_docs доступен всегда; calculator/get_date — только если их выбрал роутер
            include = set(route.tools)
            registry = default_registry(search=search_docs, now=self.now, include=include)
            messages = [
                {"role": "system", "content": TOOL_SYSTEM},
                {"role": "user", "content": prompt},
            ]
            loop = run_tool_loop(self.llm, messages, registry, max_steps=self.max_tool_steps)
            text, trace = loop.content, loop.trace
        else:
            raw = self.llm.complete(prompt, system=ANSWER_SYSTEM)
            text, trace = (raw if isinstance(raw, str) else str(raw)), []

        return RAGAnswer(
            question=question,
            answer=text,
            mode=self.mode,
            citations=make_citations(text, sources),
            rewritten_query=rewritten,
            route=route.model_dump(),
            tool_calls=trace,
            retrieved=results,
        )
