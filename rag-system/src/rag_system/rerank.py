"""Реранкинг кандидатов после гибридного поиска.

* FastEmbedReranker — cross-encoder (fastembed.rerank.cross_encoder.TextCrossEncoder),
  оценивает пару (запрос, фрагмент) целиком. Нужен extra `local`.
* LexicalReranker — лёгкий fallback без моделей: покрытие терминов запроса, совпадение
  биграмм и бонус за совпадение с заголовком секции; исходный порядок — как tie-breaker.
"""

from __future__ import annotations

import importlib.util
import itertools
import logging
from collections.abc import Sequence
from typing import Any, Protocol

from rag_system.schemas import SearchResult
from rag_system.text import tokenize

log = logging.getLogger(__name__)
DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, results: Sequence[SearchResult], top_k: int
    ) -> list[SearchResult]: ...


class NoopReranker:
    name = "none"

    def rerank(self, query: str, results: Sequence[SearchResult], top_k: int) -> list[SearchResult]:
        return list(results)[:top_k]


def _bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(itertools.pairwise(tokens))


class LexicalReranker:
    name = "lexical"

    def __init__(self, coverage_w: float = 1.0, bigram_w: float = 0.5, heading_w: float = 0.3):
        self.coverage_w = coverage_w
        self.bigram_w = bigram_w
        self.heading_w = heading_w

    def score(self, query: str, result: SearchResult) -> float:
        q = tokenize(query)
        if not q:
            return 0.0
        q_set = set(q)
        d = tokenize(result.chunk.text)
        d_set = set(d)
        coverage = len(q_set & d_set) / len(q_set)
        qb = _bigrams(q)
        bigram = len(qb & _bigrams(d)) / len(qb) if qb else 0.0
        h_set = set(tokenize(result.chunk.heading))
        heading = len(q_set & h_set) / len(q_set)
        return self.coverage_w * coverage + self.bigram_w * bigram + self.heading_w * heading

    def rerank(self, query: str, results: Sequence[SearchResult], top_k: int) -> list[SearchResult]:
        n = len(results)
        scored = []
        for rank, r in enumerate(results):
            prior = 0.05 * (n - rank) / max(n, 1)  # мягко учитываем порядок после RRF
            scored.append((self.score(query, r) + prior, rank, r))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [
            SearchResult(chunk=r.chunk, score=float(s), retriever="rerank")
            for s, _, r in scored[:top_k]
        ]


class FastEmbedReranker:
    name = "fastembed"

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL) -> None:
        self.model_name = model_name
        self._model: Any = None

    @staticmethod
    def available() -> bool:
        if importlib.util.find_spec("fastembed") is None:
            return False
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: F401
        except Exception:  # pragma: no cover - старые версии fastembed
            return False
        return True

    def _load(self) -> Any:
        if self._model is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._model = TextCrossEncoder(model_name=self.model_name)
        return self._model

    def rerank(self, query: str, results: Sequence[SearchResult], top_k: int) -> list[SearchResult]:
        if not results:
            return []
        scores = list(self._load().rerank(query, [r.chunk.text for r in results]))
        order = sorted(range(len(results)), key=lambda i: -float(scores[i]))
        return [
            SearchResult(chunk=results[i].chunk, score=float(scores[i]), retriever="rerank")
            for i in order[:top_k]
        ]


class FallbackReranker:
    """Пробует основной реранкер; при ошибке (нет сети/модели) — откатывается на запасной."""

    def __init__(self, primary: Reranker, fallback: Reranker) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}|{fallback.name}"
        self._failed = False

    def rerank(self, query: str, results: Sequence[SearchResult], top_k: int) -> list[SearchResult]:
        if not self._failed:
            try:
                return self.primary.rerank(query, results, top_k)
            except Exception as exc:  # pragma: no cover - зависит от окружения
                log.warning("Реранкер %s недоступен (%s), использую %s",
                            self.primary.name, exc, self.fallback.name)
                self._failed = True
        return self.fallback.rerank(query, results, top_k)


def get_reranker(name: str = "auto", model: str = DEFAULT_RERANK_MODEL) -> Reranker:
    name = (name or "auto").lower()
    if name == "none":
        return NoopReranker()
    if name == "lexical":
        return LexicalReranker()
    if name in {"fastembed", "cross-encoder"}:
        return FastEmbedReranker(model)
    if name == "auto":
        if FastEmbedReranker.available():
            return FallbackReranker(FastEmbedReranker(model), LexicalReranker())
        return LexicalReranker()
    raise ValueError(f"Неизвестный реранкер: {name!r} (auto | fastembed | lexical | none)")
