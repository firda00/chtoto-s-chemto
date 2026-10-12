from __future__ import annotations

import datetime as dt

import pytest

from bank_auditor.anomalies import find_duplicates, find_spikes
from bank_auditor.models import Transaction
from bank_auditor.subscriptions import burn, detect_subscriptions

PLANTED_MONTHLY = [
    "NETFLIX",
    "SPOTIFY",
    "APPLE BILL ICLOUD",
    "YANDEX PLUS",
    "PAYNET BEELINE",
    "ATLETIK FITNESS CLUB",
    "MEGOGO",
    "UZTELECOM INTERNET SHARTNOMA",
]


@pytest.mark.parametrize("merchant", PLANTED_MONTHLY)
def test_planted_subscriptions_found_with_monthly_period(subs_by_merchant, merchant):
    sub = subs_by_merchant[merchant]
    assert sub.period_label == "monthly"
    assert 28 <= sub.period_days <= 32
    assert sub.active
    assert sub.confidence >= 0.7


def test_occurrences_and_amounts(subs_by_merchant):
    assert subs_by_merchant["NETFLIX"].occurrences == 6
    assert subs_by_merchant["SPOTIFY"].occurrences == 6  # дубль не считается отдельным месяцем
    assert subs_by_merchant["MEGOGO"].occurrences == 4
    assert subs_by_merchant["ATLETIK FITNESS CLUB"].amount == 350_000
    assert subs_by_merchant["NETFLIX"].amount == pytest.approx(126_000, rel=0.02)
    nf = subs_by_merchant["NETFLIX"]
    assert nf.next_expected == nf.last_date + dt.timedelta(days=nf.period_days)


def test_forgotten_gym_is_flagged(subs_by_merchant):
    gym = subs_by_merchant["ATLETIK FITNESS CLUB"]
    assert gym.forgotten
    assert "usage_stopped" in gym.flags
    assert gym.hint.startswith("Отменить?")
    for name in ("NETFLIX", "SPOTIFY", "PAYNET BEELINE", "YANDEX PLUS"):
        assert not subs_by_merchant[name].forgotten, name


def test_trial_turned_paid(subs_by_merchant):
    megogo = subs_by_merchant["MEGOGO"]
    assert "trial_to_paid" in megogo.flags
    assert megogo.trial_amount == 1_000
    assert megogo.forgotten


def test_price_increase(subs_by_merchant):
    ya = subs_by_merchant["YANDEX PLUS"]
    assert "price_increase" in ya.flags
    assert ya.old_amount == 29_900
    assert ya.amount == 39_900
    assert ya.occurrences == 6  # два ценовых сегмента склеены в один ряд
    assert (
        "NETFLIX" in subs_by_merchant and "price_increase" not in subs_by_merchant["NETFLIX"].flags
    )


def test_duplicate_charge_flagged(subs_by_merchant, sample_report):
    sp = subs_by_merchant["SPOTIFY"]
    assert "duplicate_charge" in sp.flags
    assert [d.month for d in sp.duplicate_dates] == [7]
    dups = [a for a in sample_report.anomalies if a.kind == "duplicate"]
    assert [(a.merchant, a.date.month) for a in dups] == [("SPOTIFY", 7)]


def test_small_recurring_and_cancelled(subs_by_merchant):
    assert "small_recurring" in subs_by_merchant["APPLE BILL ICLOUD"].flags
    assert not subs_by_merchant["APPLE BILL ICLOUD"].forgotten
    ivi = subs_by_merchant["IVI"]
    assert not ivi.active and "inactive" in ivi.flags and not ivi.forgotten


def test_no_false_positives_on_everyday_spending(subs_by_merchant):
    for noisy in (
        "KORZINKA CHILONZOR",
        "MAKRO YUNUSOBOD",
        "YANDEX GO TAXI",
        "EVOS",
        "PEREVOD NA KARTU",
        "KARIMOV A A",
        "ZIYO NUR",
        "HUDUDIY ELEKTR TARMOQLARI",
    ):
        assert noisy not in subs_by_merchant, noisy


def test_burn_math(sample_report):
    monthly, annual = burn(sample_report.subscriptions)
    assert monthly == sample_report.monthly_burn
    assert annual == pytest.approx(monthly * 12, rel=0.01)
    inactive_cost = sum(s.monthly_cost for s in sample_report.subscriptions if not s.active)
    assert inactive_cost > 0  # IVI есть, но в нагрузку не входит
    forgotten = {s.merchant for s in sample_report.subscriptions if s.forgotten}
    assert forgotten == {"ATLETIK FITNESS CLUB", "MEGOGO"}


def _t(day: dt.date, amount: float, merchant: str = "GYM") -> Transaction:
    return Transaction(
        date=day,
        amount=amount,
        description=merchant,
        merchant_normalized=merchant,
        category="health",
    )


def test_weekly_and_no_related_activity():
    start = dt.date(2026, 1, 5)
    txns = [_t(start + dt.timedelta(days=7 * i), 50_000, "SWIM CLUB") for i in range(8)]
    (sub,) = detect_subscriptions(txns)
    assert sub.period_label == "weekly" and sub.period_days == 7
    assert "no_related_activity" in sub.flags and sub.forgotten
    assert sub.monthly_cost == pytest.approx(50_000 * 30.44 / 7, rel=1e-3)


def test_yearly_with_two_exact_charges():
    txns = [
        _t(dt.date(2025, 3, 1), 499_000, "ANTIVIRUS"),
        _t(dt.date(2026, 3, 1), 499_000, "ANTIVIRUS"),
    ]
    (sub,) = detect_subscriptions(txns, as_of=dt.date(2026, 6, 1))
    assert sub.period_label == "yearly"
    assert sub.annual_cost == pytest.approx(499_000, rel=0.01)


def test_irregular_charges_are_not_subscriptions():
    days = [1, 3, 17, 18, 40, 77, 80, 130]
    txns = [_t(dt.date(2026, 1, 1) + dt.timedelta(days=d), 100_000, "SHOP") for d in days]
    assert detect_subscriptions(txns) == []


def test_spike_detection(sample_report):
    spikes = [a for a in sample_report.anomalies if a.kind == "spike"]
    assert [(a.merchant, a.amount) for a in spikes] == [("TEXNOMART SERGELI", 8_900_000)]


def test_duplicates_and_spikes_helpers():
    d = dt.date(2026, 5, 1)
    txns = [_t(d, 10_000, "CAFE"), _t(d, 10_000, "CAFE"), _t(d, 12_000, "CAFE")]
    assert len(find_duplicates(txns)) == 1
    assert find_spikes(txns) == []  # слишком мало данных для статистики
