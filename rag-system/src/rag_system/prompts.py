"""Промпты и форматирование контекста — общие для naive и modular пайплайнов."""

from __future__ import annotations

from collections.abc import Sequence

from rag_system.schemas import SearchResult

ANSWER_SYSTEM = (
    "You are a precise assistant for a company knowledge base. Answer ONLY using the numbered "
    "context fragments. Cite every fact with its fragment number in square brackets, e.g. [1] "
    "or [2][3]. If the context does not contain the answer, say so honestly. "
    "Answer in the same language as the question, briefly (1-3 sentences)."
)


def format_context(results: Sequence[SearchResult], start: int = 1) -> str:
    blocks = []
    for n, r in enumerate(results, start=start):
        header = f"[{n}] {r.chunk.source}"
        if r.chunk.heading:
            header += f" — {r.chunk.heading}"
        blocks.append(f"{header}\n{r.chunk.text.strip()}")
    return "\n\n".join(blocks)


def build_answer_prompt(question: str, results: Sequence[SearchResult]) -> str:
    return (
        f"<context>\n{format_context(results)}\n</context>\n\n"
        f"<question>{question}</question>\n\n"
        "Answer the question using only the context above and cite fragments like [1]."
    )


REWRITE_SYSTEM = (
    "You rewrite user questions into short standalone search queries for a hybrid "
    "(BM25 + vector) search engine. Keep the original language, keep names, numbers and key "
    "terms, drop greetings and filler. Return ONLY the rewritten query, nothing else."
)


def build_rewrite_prompt(question: str) -> str:
    return f"<query>{question}</query>\nRewritten query:"
