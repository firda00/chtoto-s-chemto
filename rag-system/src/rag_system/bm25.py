"""Лексический поиск BM25 (rank-bm25) поверх тех же чанков, что лежат в VectorStore."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from rag_system.schemas import Chunk, SearchResult
from rag_system.text import tokenize


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        corpus = [tokenize(c.text) or ["∅"] for c in self.chunks]
        self._bm25 = BM25Okapi(corpus, k1=k1, b=b) if corpus else None

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        terms = tokenize(query)
        if self._bm25 is None or not terms or k <= 0:
            return []
        scores = np.asarray(self._bm25.get_scores(terms), dtype=np.float64)
        order = np.argsort(-scores, kind="stable")[:k]
        return [
            SearchResult(chunk=self.chunks[i], score=float(scores[i]), retriever="bm25")
            for i in order
            if scores[i] > 0
        ]
