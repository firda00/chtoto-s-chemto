"""rag_system — Naive RAG и Modular RAG поверх единого VectorStore-протокола."""

from rag_system.schemas import Chunk, Citation, Document, RAGAnswer, SearchResult

__all__ = ["Chunk", "Citation", "Document", "RAGAnswer", "SearchResult"]
__version__ = "0.1.0"
