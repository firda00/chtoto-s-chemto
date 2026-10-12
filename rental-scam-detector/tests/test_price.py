from __future__ import annotations

import pytest

from scam_detector.text_features import price_deviation, to_usd


def test_price_deviation_vs_district_median(districts):
    # Мирабад: медиана 9.5 USD/м², std 2.0
    p = price_deviation(285, "USD", 60, "Мирабадский район", districts)
    assert p.district_known
    assert p.usd_per_m2 == pytest.approx(4.75)
    assert p.ratio == pytest.approx(0.5)
    assert p.z == pytest.approx(-2.375)
    assert p.too_good


def test_market_price_is_not_too_good(districts):
    p = price_deviation(460, "USD", 62, "Yunusobod", districts)
    assert 0.9 < p.ratio < 1.1 and not p.too_good


def test_uzs_conversion_and_aliases(districts, monkeypatch):
    monkeypatch.setenv("SCAM_UZS_PER_USD", "12500")
    assert to_usd(1_250_000, "UZS") == pytest.approx(100)
    assert to_usd(100, "usd") == 100
    p = price_deviation(3_250_000, "UZS", 40, "Chilonzor tumani", districts)
    assert p.district_known and p.usd_per_m2 == pytest.approx(6.5)
    assert districts.canonical("Мирзо-Улугбек") == districts.canonical("Mirzo Ulugbek")


def test_unknown_district_falls_back_to_city(districts):
    p = price_deviation(500, "USD", 70, "Атлантида", districts)
    assert not p.district_known
    assert p.median_usd_m2 == districts.city.median_usd_m2
