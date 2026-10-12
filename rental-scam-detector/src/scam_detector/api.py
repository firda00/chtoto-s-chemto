"""FastAPI: POST /check → ScamReport."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from .config import corpus_dir as default_corpus_dir
from .config import model_path as default_model_path
from .features import FeatureContext
from .models import Listing, ScamReport
from .predict import ScamDetector, build_context


def create_app(
    model_path: str | Path | None = None,
    corpus_dir: str | Path | None = None,
    detector: ScamDetector | None = None,
    context: FeatureContext | None = None,
) -> FastAPI:
    """Модель и корпус (для поиска дубликатов фото/текста/телефонов) грузятся один раз."""
    app = FastAPI(title="Rental Scam Detector", version="0.1.0")
    state: dict = {"detector": detector, "context": context}

    def get_detector() -> ScamDetector:
        if state["detector"] is None:
            try:
                state["detector"] = ScamDetector.load(model_path or default_model_path())
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        return state["detector"]

    def get_context() -> FeatureContext:
        if state["context"] is None:
            state["context"] = build_context(corpus_dir or default_corpus_dir())
        return state["context"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/check", response_model=ScamReport)
    def check(listing: Listing) -> ScamReport:
        return get_detector().report(listing, get_context())

    return app


app = create_app()
