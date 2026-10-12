from __future__ import annotations

from pathlib import Path

import pytest

from mlops_tracker.evals.dataset import Dataset
from mlops_tracker.tracker import Tracker

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "evals"


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Тесты всегда офлайн: никакого ключа API, git не вызывается, судья — MockLLM."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("MLT_DISABLE_GIT", "1")


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "mlruns" / "mlt.db"


@pytest.fixture
def tracker(db_path: Path) -> Tracker:
    return Tracker(db_path)


@pytest.fixture
def small_dataset() -> Dataset:
    return Dataset.from_records(
        [
            {"id": "a", "input": "2+2", "expected": "4"},
            {"id": "b", "input": "столица Франции", "expected": "Париж"},
            {"id": "c", "input": "hello", "expected": "world"},
        ],
        name="small",
    )
