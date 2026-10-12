"""Pydantic-схемы, общие для всех модулей."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Исходный документ (файл) до разбиения на фрагменты."""

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Фрагмент документа — единица индексации и поиска."""

    id: str
    doc_id: str
    text: str
    collection: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", self.doc_id))

    @property
    def heading(self) -> str:
        return str(self.metadata.get("heading", ""))


class SearchResult(BaseModel):
    chunk: Chunk
    score: float
    retriever: str = "vector"  # vector | bm25 | rrf | rerank


class Citation(BaseModel):
    n: int
    chunk_id: str
    source: str
    heading: str = ""
    snippet: str = ""


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str = ""


class RAGAnswer(BaseModel):
    question: str
    answer: str
    mode: str
    citations: list[Citation] = Field(default_factory=list)
    rewritten_query: str | None = None
    route: dict[str, Any] | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    retrieved: list[SearchResult] = Field(default_factory=list, exclude=True)
