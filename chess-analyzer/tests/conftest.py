from pathlib import Path

import pytest

from chess_analyzer.analysis import analyse_game
from chess_analyzer.engine import FakeEngine
from chess_analyzer.pgn import load_games

DATA = Path(__file__).resolve().parent.parent / "data" / "games"


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    """Тесты не ходят в сеть: убираем ключ LLM, чтобы ``auto`` не выбрал OpenAI."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA


@pytest.fixture(scope="session")
def missed_mate_analysis():
    game = load_games(DATA / "missed_mate.pgn")[0]
    return analyse_game(game, FakeEngine(max_depth=2))


@pytest.fixture(scope="session")
def sample_analyses():
    return [analyse_game(g, FakeEngine(max_depth=2)) for g in load_games(DATA / "sample.pgn")]
