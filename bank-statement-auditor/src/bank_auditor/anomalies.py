"""Аномалии: двойные списания и необычные всплески трат."""

from __future__ import annotations

import statistics
from collections import defaultdict

from .models import CATEGORY_LABELS_RU, Anomaly, Transaction

SPIKE_EXCLUDED = frozenset({"transfers", "cash", "salary"})


def _fmt(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def find_duplicates(txns: list[Transaction]) -> list[Anomaly]:
    """Одинаковые списания: тот же день + тот же мерчант + та же сумма."""
    groups: dict[tuple, list[Transaction]] = defaultdict(list)
    for t in txns:
        if t.direction == "debit":
            groups[(t.date, t.merchant_normalized, round(t.amount, 2), t.currency)].append(t)
    out: list[Anomaly] = []
    for (date, merchant, amount, currency), items in sorted(groups.items()):
        if len(items) > 1:
            out.append(
                Anomaly(
                    kind="duplicate",
                    date=date,
                    merchant=merchant,
                    amount=amount,
                    currency=currency,
                    category=items[0].category,
                    details=f"{len(items)} одинаковых списания по {_fmt(amount)} {currency} "
                    f"в один день — возможно, двойное списание",
                )
            )
    return out


def find_spikes(
    txns: list[Transaction], z_threshold: float = 6.0, min_ratio: float = 4.0, min_count: int = 8
) -> list[Anomaly]:
    """Робастный z-score (медиана/MAD) внутри категории; для малых категорий — по всем тратам."""
    debits = [t for t in txns if t.direction == "debit" and t.category not in SPIKE_EXCLUDED]
    if len(debits) < min_count:
        return []
    by_cat: dict[str, list[Transaction]] = defaultdict(list)
    for t in debits:
        by_cat[t.category or "other"].append(t)
    all_amounts = [t.amount for t in debits]

    out: list[Anomaly] = []
    for cat, items in by_cat.items():
        own = len(items) >= min_count
        pool = [t.amount for t in items] if own else all_amounts
        scope = f"в категории «{CATEGORY_LABELS_RU.get(cat, cat)}»" if own else "по всем тратам"
        med = statistics.median(pool)
        mad = statistics.median(abs(a - med) for a in pool) or med * 0.1 or 1.0
        for t in items:
            z = 0.6745 * (t.amount - med) / mad
            if z >= z_threshold and t.amount >= min_ratio * med:
                out.append(
                    Anomaly(
                        kind="spike",
                        date=t.date,
                        merchant=t.merchant_normalized,
                        amount=t.amount,
                        currency=t.currency,
                        category=cat,
                        details=f"в {t.amount / med:.1f}× выше типичной траты {scope} "
                        f"(медиана {_fmt(med)})",
                    )
                )
    return sorted(out, key=lambda a: a.date)


def find_anomalies(txns: list[Transaction]) -> list[Anomaly]:
    return find_duplicates(txns) + find_spikes(txns)
