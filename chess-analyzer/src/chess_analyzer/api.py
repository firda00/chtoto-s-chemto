"""FastAPI-сервис: ``POST /analyze`` (PGN-текст) → JSON-отчёт.

Запуск: ``uv run uvicorn chess_analyzer.api:app --reload``
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .analysis import GameAnalysis, MoveClass, analyse_game
from .engine import make_engine
from .explain import explain_game
from .llm import get_llm
from .pgn import PGNError, load_games

app = FastAPI(title="Chess Match Analyzer", version=__version__)

MAX_GAMES = 20


class AnalyzeRequest(BaseModel):
    pgn: str = Field(..., min_length=1, description="PGN-текст (одна или несколько партий)")
    engine: Literal["auto", "fake", "stockfish", "api"] = "auto"
    depth: int = Field(12, ge=1, le=30)
    lang: Literal["ru", "en"] = "ru"
    llm: Literal["auto", "openai", "mock", "none"] = "auto"
    min_class: MoveClass = MoveClass.MISTAKE


class AnalyzeResponse(BaseModel):
    games: list[GameAnalysis]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    try:
        games = load_games(req.pgn if "\n" in req.pgn else req.pgn + "\n")
    except PGNError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if len(games) > MAX_GAMES:
        raise HTTPException(status_code=413, detail=f"Не более {MAX_GAMES} партий за запрос")
    try:
        engine = make_engine(req.engine)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    llm = get_llm(req.llm)
    try:
        analyses = [analyse_game(g, engine, depth=req.depth) for g in games]
    finally:
        engine.close()
    for ga in analyses:
        explain_game(ga, llm, lang=req.lang, min_class=req.min_class)
    return AnalyzeResponse(games=analyses)
