from __future__ import annotations

from pathlib import Path

import pytest

from scam_detector.predict import ScamDetector, build_context
from scam_detector.synth import make_dataset
from scam_detector.text_features import DistrictTable
from scam_detector.train import train_model

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"


@pytest.fixture(scope="session")
def districts() -> DistrictTable:
    return DistrictTable.load(ROOT / "data" / "district_prices.csv")


@pytest.fixture(scope="session")
def small_dataset(districts):
    return make_dataset(n=600, scam_ratio=0.3, seed=7, districts=districts)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, small_dataset):
    """Модель для тестов обучается здесь, в tmp — модель из репозитория не используется."""
    out = tmp_path_factory.mktemp("model")
    return train_model(small_dataset, out, iterations=150, seed=7)


@pytest.fixture(scope="session")
def detector(trained) -> ScamDetector:
    return ScamDetector.load(trained.model_path)


@pytest.fixture(scope="session")
def samples_context(districts):
    return build_context(SAMPLES, districts=districts)
