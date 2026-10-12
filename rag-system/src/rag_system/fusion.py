"""Reciprocal Rank Fusion (Cormack et al., 2009).

score(d) = Σ_i  w_i / (k + rank_i(d))

Не требует калибровки шкал BM25 и косинуса — работает только с рангами,
поэтому это стандартный способ склеить лексический и векторный поиск.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_system.schemas import SearchResult


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchResult]],
    k: int = 60,
    weights: Sequence[float] | None = None,
    top_n: int | None = None,
) -> list[SearchResult]:
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights и rankings должны быть одной длины")
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    items: dict[str, SearchResult] = {}
    order = 0
    for w, ranking in zip(weights, rankings, strict=True):
        seen_here: set[str] = set()
        for rank, res in enumerate(ranking, start=1):
            cid = res.chunk.id
            if cid in seen_here:  # дубль в одном списке не должен давать двойной бонус
                continue
            seen_here.add(cid)
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
            if cid not in items:
                items[cid] = res
                first_seen[cid] = order
                order += 1
    ranked = sorted(scores, key=lambda cid: (-scores[cid], first_seen[cid]))
    if top_n is not None:
        ranked = ranked[:top_n]
    return [SearchResult(chunk=items[c].chunk, score=scores[c], retriever="rrf") for c in ranked]
