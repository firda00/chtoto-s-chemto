"""HTTP API на FastAPI.

    GET  /health                → состояние индекса и компонентов
    POST /ask    {question, mode, k}
    POST /ingest {path, chunker, reset}
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_system.chunking import get_chunker
from rag_system.factory import Runtime
from rag_system.schemas import RAGAnswer


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    mode: Literal["naive", "modular"] = "modular"
    k: int | None = Field(default=None, ge=1, le=20)


class IngestRequest(BaseModel):
    path: str
    chunker: Literal["markdown", "fixed"] = "markdown"
    reset: bool = True


def create_app(runtime_factory: Callable[[], Runtime] | None = None) -> FastAPI:
    app = FastAPI(title="RAG System", version="0.1.0")
    state: dict[str, Runtime] = {}
    factory = runtime_factory or Runtime.open

    def runtime() -> Runtime:
        if "rt" not in state:
            state["rt"] = factory()
        return state["rt"]

    @app.get("/health")
    def health() -> dict[str, object]:
        rt = runtime()
        return {
            "status": "ok",
            "chunks": rt.store.count(),
            "collections": rt.store.collections(),
            "store": type(rt.store).__name__,
            "embedder": rt.embedder.name,
            "reranker": rt.reranker.name,
            "llm": rt.llm.name,
        }

    @app.post("/ask", response_model=RAGAnswer)
    def ask(req: AskRequest) -> RAGAnswer:
        rt = runtime()
        if rt.store.count() == 0:
            raise HTTPException(409, "index is empty: POST /ingest first")
        return rt.pipeline(req.mode, req.k).answer(req.question)

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict[str, int]:
        try:
            return runtime().ingest(req.path, get_chunker(req.chunker), reset=req.reset)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    return app
