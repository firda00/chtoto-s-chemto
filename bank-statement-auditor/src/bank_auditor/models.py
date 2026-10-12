"""Pydantic-схемы: транзакция, подписка, аномалия, итоговый отчёт."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field

Direction = Literal["debit", "credit"]

CATEGORIES: tuple[str, ...] = (
    "food",
    "transport",
    "subscriptions",
    "utilities",
    "health",
    "entertainment",
    "shopping",
    "transfers",
    "cash",
    "salary",
    "other",
)

CATEGORY_LABELS_RU: dict[str, str] = {
    "food": "Еда и продукты",
    "transport": "Транспорт",
    "subscriptions": "Подписки и софт",
    "utilities": "Связь и коммуналка",
    "health": "Здоровье и спорт",
    "entertainment": "Развлечения",
    "shopping": "Покупки",
    "transfers": "Переводы",
    "cash": "Наличные",
    "salary": "Зарплата",
    "other": "Прочее",
}


class Transaction(BaseModel):
    """Одна операция по выписке. amount всегда положительный, знак — в direction."""

    date: dt.date
    amount: float = Field(ge=0)
    currency: str = "UZS"
    description: str
    merchant_normalized: str = ""
    category: str | None = None
    direction: Direction = "debit"
    source: str = ""

    @property
    def signed_amount(self) -> float:
        return -self.amount if self.direction == "debit" else self.amount


class ParseResult(BaseModel):
    """Результат парсинга файла + метаданные автоопределения."""

    transactions: list[Transaction]
    format_name: str
    encoding: str | None = None
    delimiter: str | None = None
    date_format: str | None = None
    column_mapping: dict[str, str] = Field(default_factory=dict)
    skipped_rows: int = 0


class Subscription(BaseModel):
    """Регулярное списание, найденное детектором."""

    merchant: str
    category: str | None = None
    amount: float  # текущая (последняя типичная) сумма списания
    currency: str = "UZS"
    period_days: int
    period_label: str  # weekly / biweekly / monthly / quarterly / yearly
    occurrences: int
    first_date: dt.date
    last_date: dt.date
    next_expected: dt.date
    monthly_cost: float
    annual_cost: float
    confidence: float = Field(ge=0, le=1)
    active: bool = True
    amount_history: list[float] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    forgotten: bool = False
    forgotten_score: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    hint: str = ""
    old_amount: float | None = None  # цена до повышения (если было)
    trial_amount: float | None = None  # «пробное» списание перед платными
    duplicate_dates: list[dt.date] = Field(default_factory=list)


class Anomaly(BaseModel):
    kind: Literal["duplicate", "spike"]
    date: dt.date
    merchant: str
    amount: float
    currency: str = "UZS"
    category: str | None = None
    details: str = ""


class MerchantStat(BaseModel):
    merchant: str
    category: str | None = None
    total: float
    count: int


class AuditReport(BaseModel):
    source_file: str
    format_name: str
    period_start: dt.date
    period_end: dt.date
    currency: str
    n_transactions: int
    total_spent: float
    total_income: float
    spending_by_category: dict[str, float]
    monthly_spending: dict[str, float]
    top_merchants: list[MerchantStat]
    subscriptions: list[Subscription]
    monthly_burn: float
    annual_burn: float
    forgotten_annual_savings: float
    anomalies: list[Anomaly]
    uncategorized: int = 0
    other_currency_transactions: int = 0
    transactions: list[Transaction] = Field(default_factory=list, exclude=True)
