from __future__ import annotations

import pytest

from scam_detector.features import FEATURE_NAMES
from scam_detector.synth import generate_listings, make_dataset


def test_dataset_shape_and_labels(small_dataset):
    df = small_dataset
    assert df.shape == (600, len(FEATURE_NAMES) + 3)
    assert list(df.columns) == ["listing_id", "lang", *FEATURE_NAMES, "label"]
    assert set(df["label"].unique()) == {0, 1}
    assert 0.25 < df["label"].mean() < 0.35  # 0.3 плюс шум разметки
    assert df["listing_id"].is_unique
    assert df[FEATURE_NAMES].notna().all().all()


def test_scam_signals_correlate_with_label(small_dataset):
    g = small_dataset.groupby("label").mean(numeric_only=True)
    for col in ("prepayment_hits", "urgency_hits", "owner_abroad_hits", "photo_dup_count"):
        assert g.loc[1, col] > g.loc[0, col], col
    assert g.loc[1, "price_ratio_to_median"] < g.loc[0, "price_ratio_to_median"]


def test_generation_is_deterministic_and_ratio_controllable(districts):
    a = make_dataset(n=120, scam_ratio=0.5, seed=1, districts=districts)
    b = make_dataset(n=120, scam_ratio=0.5, seed=1, districts=districts)
    assert a.equals(b)
    items = generate_listings(n=100, scam_ratio=0.1, seed=3, districts=districts)
    assert sum(x.listing.id.startswith("S") for x in items) == 10


def test_invalid_ratio(districts):
    with pytest.raises(ValueError):
        generate_listings(n=10, scam_ratio=1.0, districts=districts)
