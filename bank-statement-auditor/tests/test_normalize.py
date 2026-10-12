from __future__ import annotations

import pytest

from bank_auditor.normalize import (
    merchant_root,
    normalize_currency,
    normalize_merchant,
    parse_amount,
    translit_lat_to_ru,
    translit_ru_to_lat,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("125 000,00", 125000.0),
        ("125 000,50", 125000.5),
        ("125,000.00", 125000.0),
        ("1.234.567,89", 1234567.89),
        ("1,234,567.89", 1234567.89),
        ("-1 234.5", -1234.5),
        ("−450,00", -450.0),
        ("(99.90)", -99.9),
        ("150.00-", -150.0),
        ("+12 500 000.00", 12500000.0),
        ("UZS 15 000", 15000.0),
        ("$9.99", 9.99),
        ("9,99", 9.99),
        ("1,234", 1234.0),
        ("1'250'000", 1250000.0),
        ("12.5", 12.5),
        (42, 42.0),
        ("", None),
        ("nan", None),
        ("—", None),
        ("abc", None),
    ],
)
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("UZS", "UZS"),
        ("сум", "UZS"),
        ("so'm", "UZS"),
        ("so’m", "UZS"),
        ("860", "UZS"),
        ("$", "USD"),
        ("usd", "USD"),
        ("руб.", "RUB"),
        ("€", "EUR"),
        ("CHF", "CHF"),
        ("", "UZS"),
        (None, "UZS"),
        ("???", "UZS"),
    ],
)
def test_normalize_currency(raw, expected):
    assert normalize_currency(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("NETFLIX.COM 4169****7788 RRN 12345678 AMSTERDAM NL", "NETFLIX"),
        ("SPOTIFY ID 30458812 STOCKHOLM SE", "SPOTIFY"),
        ("YANDEX*PLUS MOSCOW RU", "YANDEX PLUS"),
        ("ЯНДЕКС.ПЛЮС МОСКВА", "YANDEX PLUS"),
        ("APPLE.COM/BILL ICLOUD CUPERTINO US", "APPLE BILL ICLOUD"),
        ("OOO KORZINKA CHILONZOR TOSHKENT", "KORZINKA CHILONZOR"),
        ("Покупка: ООО «Корзинка» Ташкент 05.04.2026 14:32", "KORZINKA"),
        ("POS PURCHASE EVOS 8600 12** **** 3456", "EVOS"),
        ("PAYNET BEELINE 998901234567", "PAYNET BEELINE"),
        ("MEGOGO.NET LIMASSOL CY", "MEGOGO"),
        ("HUDUDGAZ TOSHKENT TO'LOV", "HUDUDGAZ TO'LOV"),
        ("ПЕРЕВОД НА КАРТУ 8600****1122", "PEREVOD NA KARTU"),
        ("TXN#A1B2C3D4E5 UBER *TRIP", "UBER TRIP"),
    ],
)
def test_normalize_merchant(raw, expected):
    assert normalize_merchant(raw) == expected


def test_same_merchant_with_different_ids_groups_together():
    a = normalize_merchant("SPOTIFY ID 11112222 STOCKHOLM SE")
    b = normalize_merchant("SPOTIFY ID 99998888 STOCKHOLM SE")
    assert a == b == "SPOTIFY"


def test_translit_roundtrip_basic():
    assert translit_ru_to_lat("Яндекс Плюс") == "YANDEKS PLYUS"
    assert translit_ru_to_lat("Щука Ёж") == "SCHUKA YOZH"
    assert translit_lat_to_ru("Korzinka") == "КОРЗИНКА"
    assert translit_lat_to_ru("Chorsu bozori") == "ЧОРСУ БОЗОРИ"
    assert translit_ru_to_lat(translit_lat_to_ru("MAKRO")) == "MAKRO"


def test_merchant_root():
    assert merchant_root("ATLETIK FITNESS CLUB") == "ATLETIK"
    assert merchant_root("IVI") == "IVI"
