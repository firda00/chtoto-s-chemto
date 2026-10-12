"""Сквозной аудит: парсинг → категоризация → подписки → аномалии → AuditReport."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from .anomalies import find_anomalies
from .categorize import Categorizer
from .llm import LLM
from .models import AuditReport, MerchantStat, Transaction
from .parsers import load_statement
from .subscriptions import DetectorConfig, burn, detect_subscriptions


def build_report(
    txns: list[Transaction],
    *,
    source_file: str = "",
    format_name: str = "",
    config: DetectorConfig | None = None,
    top_n: int = 10,
) -> AuditReport:
    if not txns:
        raise ValueError("В выписке нет ни одной операции")
    currency = Counter(t.currency for t in txns).most_common(1)[0][0]
    main = [t for t in txns if t.currency == currency]

    df = pd.DataFrame(
        [
            {
                "date": t.date,
                "month": t.date.strftime("%Y-%m"),
                "amount": t.amount,
                "direction": t.direction,
                "category": t.category or "other",
                "merchant": t.merchant_normalized,
            }
            for t in main
        ]
    )
    spent = df[df["direction"] == "debit"]
    income = df[df["direction"] == "credit"]

    by_cat = spent.groupby("category")["amount"].sum().sort_values(ascending=False)
    monthly = spent.groupby("month")["amount"].sum().sort_index()
    merchants = (
        spent[~spent["category"].isin(["transfers", "cash"])]
        .groupby("merchant")
        .agg(total=("amount", "sum"), count=("amount", "size"), category=("category", "first"))
        .sort_values("total", ascending=False)
        .head(top_n)
    )

    subs = detect_subscriptions(main, config=config)
    monthly_burn, annual_burn = burn(subs)
    savings = round(sum(s.annual_cost for s in subs if s.forgotten), 2)

    return AuditReport(
        source_file=source_file,
        format_name=format_name,
        period_start=min(t.date for t in txns),
        period_end=max(t.date for t in txns),
        currency=currency,
        n_transactions=len(txns),
        total_spent=round(float(spent["amount"].sum()), 2),
        total_income=round(float(income["amount"].sum()), 2),
        spending_by_category={k: round(float(v), 2) for k, v in by_cat.items()},
        monthly_spending={k: round(float(v), 2) for k, v in monthly.items()},
        top_merchants=[
            MerchantStat(
                merchant=str(name),
                category=row["category"],
                total=round(float(row["total"]), 2),
                count=int(row["count"]),
            )
            for name, row in merchants.iterrows()
        ],
        subscriptions=subs,
        monthly_burn=monthly_burn,
        annual_burn=annual_burn,
        forgotten_annual_savings=savings,
        anomalies=find_anomalies(main),
        uncategorized=sum(1 for t in txns if (t.category or "other") == "other"),
        other_currency_transactions=len(txns) - len(main),
        transactions=txns,
    )


def run_audit(
    path: str | Path,
    *,
    llm: LLM | None = None,
    default_currency: str = "UZS",
    config: DetectorConfig | None = None,
) -> AuditReport:
    parsed = load_statement(path, default_currency=default_currency)
    txns = Categorizer(llm=llm).categorize(parsed.transactions)
    return build_report(
        txns, source_file=Path(path).name, format_name=parsed.format_name, config=config
    )
