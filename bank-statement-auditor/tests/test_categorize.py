from __future__ import annotations

import datetime as dt

import pytest

from bank_auditor.categorize import Categorizer, CategoryGuess, categorize, rule_category
from bank_auditor.llm import MockLLM
from bank_auditor.models import Transaction
from bank_auditor.normalize import normalize_merchant


def tx(desc: str, amount: float = 50_000, direction: str = "debit") -> Transaction:
    return Transaction(
        date=dt.date(2026, 4, 1),
        amount=amount,
        description=desc,
        merchant_normalized=normalize_merchant(desc),
        direction=direction,
    )


@pytest.mark.parametrize(
    ("desc", "category"),
    [
        # EN
        ("NETFLIX.COM AMSTERDAM NL", "subscriptions"),
        ("SPOTIFY ID 30458812 STOCKHOLM SE", "subscriptions"),
        ("APPLE.COM/BILL ICLOUD CUPERTINO US", "subscriptions"),
        ("YANDEX GO TAXI", "transport"),
        ("ATM KAPITALBANK CHILONZOR", "cash"),
        ("COFFEE HOUSE MAGIC CITY", "food"),
        ("MAGIC CITY PARK", "entertainment"),
        # RU
        ("ЯНДЕКС.ПЛЮС МОСКВА", "subscriptions"),
        ("Аптека 999 Ташкент", "health"),
        ("КАФЕ РОХАТ", "food"),
        ("ПЕРЕВОД НА КАРТУ 8600****1122", "transfers"),
        ("СНЯТИЕ НАЛИЧНЫХ БАНКОМАТ 0451", "cash"),
        ("Оплата коммунальных услуг", "utilities"),
        # UZ
        ("DORIXONA GRAND PHARM", "health"),
        ("P2P O'TKAZMA HUMO 9860****3344", "transfers"),
        ("HUDUDGAZ TOSHKENT TO'LOV", "utilities"),
        ("PAYNET BEELINE 998901234567", "utilities"),
        ("OQTEPA LAVASH", "food"),
        ("ATLETIK FITNESS CLUB TOSHKENT", "health"),
        ("UZUM MARKET", "shopping"),
        ("UZUM TEZKOR DOSTAVKA", "food"),
    ],
)
def test_rule_categories(desc, category):
    assert rule_category(tx(desc)) == category


def test_credit_rules():
    assert rule_category(tx("ЗАРПЛАТА ООО DIGITAL SOLUTIONS", 12_500_000, "credit")) == "salary"
    assert rule_category(tx("OYLIK ISH HAQI", 9_000_000, "credit")) == "salary"
    assert rule_category(tx("P2P PEREVOD OT 8600****5566", 300_000, "credit")) == "transfers"
    # зачисление от магазина (возврат) не должно стать «едой»
    assert rule_category(tx("KORZINKA REFUND", 30_000, "credit")) == "transfers"


def test_unknown_merchant_without_llm_is_other():
    assert rule_category(tx("IP KARIMOV A.A.")) is None
    assert categorize([tx("IP KARIMOV A.A.")])[0].category == "other"


def test_llm_fallback_with_mock_and_cache():
    llm = MockLLM(
        rules={
            "ZIYO NUR": {"category": "shopping", "confidence": 0.9, "reason": "книжный магазин"},
            "KARIMOV": {"category": "food", "confidence": 0.3, "reason": "не уверен"},
        }
    )
    cat = Categorizer(llm=llm)
    out = cat.categorize(
        [tx("OOO ZIYO NUR"), tx("OOO ZIYO NUR", 70_000), tx("IP KARIMOV A.A."), tx("NETFLIX.COM")]
    )
    assert [t.category for t in out] == ["shopping", "shopping", "other", "subscriptions"]
    assert cat.llm_calls == 2  # ZIYO NUR закеширован, NETFLIX решён правилами
    # structured output: в запрос уходит JSON-схема Pydantic-модели
    assert llm.calls[0]["json_schema"] == CategoryGuess.model_json_schema()
    assert "ZIYO NUR" in llm.calls[0]["prompt"]


def test_llm_invalid_output_falls_back_to_other():
    llm = MockLLM(rules={"ZIYO": {"category": "crypto", "confidence": 1.0}})
    cat = Categorizer(llm=llm)
    assert cat.categorize_one(tx("OOO ZIYO NUR")) == "other"
    assert cat.llm_errors == 1


def test_llm_exception_does_not_crash():
    class Boom:
        def complete(self, prompt, system=None, json_schema=None):
            raise ConnectionError("нет сети")

    cat = Categorizer(llm=Boom())
    assert cat.categorize_one(tx("OOO ZIYO NUR")) == "other"


def test_sample_statement_mostly_categorized(sample_txns):
    other = [t for t in sample_txns if t.category == "other"]
    assert len(other) / len(sample_txns) < 0.03
    by_merchant = {t.merchant_normalized: t.category for t in sample_txns}
    assert by_merchant["NETFLIX"] == "subscriptions"
    assert by_merchant["YANDEX PLUS"] == "subscriptions"
    assert by_merchant["ZARPLATA DIGITAL SOLUTIONS"] == "salary"
    assert by_merchant["KAFE ROKHAT"] == "food"
