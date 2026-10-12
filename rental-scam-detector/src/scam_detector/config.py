"""Пути и настройки по умолчанию (переопределяются переменными окружения)."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def model_path() -> Path:
    return _env_path("SCAM_MODEL_PATH", PROJECT_ROOT / "models" / "scam.cbm")


def districts_csv() -> Path:
    return _env_path("SCAM_DISTRICTS_CSV", DATA_DIR / "district_prices.csv")


def corpus_dir() -> Path:
    return _env_path("SCAM_CORPUS_DIR", DATA_DIR / "samples")


def uzs_per_usd() -> float:
    return float(os.environ.get("SCAM_UZS_PER_USD", "12700"))


# Порог близости перцептивных хэшей (бит из 64): max(pHash, dHash) <= порога → дубликат.
PHOTO_HAMMING_THRESHOLD = int(os.environ.get("SCAM_PHOTO_THRESHOLD", "10"))
