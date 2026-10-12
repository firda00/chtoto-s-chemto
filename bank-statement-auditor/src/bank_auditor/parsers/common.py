"""Общая логика парсеров: распознавание колонок, форматов дат, сборка транзакций."""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable, Sequence

import pandas as pd

from ..models import Transaction
from ..normalize import normalize_currency, normalize_merchant, parse_amount

# логическое поле -> синонимы заголовков (RU / EN / UZ), сравнение по нормализованному виду
COLUMN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "date": (
        "date",
        "posted date",
        "posting date",
        "transaction date",
        "value date",
        "txn date",
        "дата",
        "дата операции",
        "дата проводки",
        "дата транзакции",
        "sana",
        "operatsiya sanasi",
    ),
    "description": (
        "description",
        "details",
        "narrative",
        "merchant",
        "payee",
        "memo",
        "transaction",
        "операция",
        "описание",
        "назначение",
        "назначение платежа",
        "описание операции",
        "детали",
        "контрагент",
        "tavsif",
        "izoh",
        "operatsiya",
        "to'lov maqsadi",
    ),
    "amount": ("amount", "sum", "сумма", "сумма операции", "summa", "miqdor"),
    "debit": (
        "debit",
        "withdrawal",
        "withdrawals",
        "money out",
        "paid out",
        "дебет",
        "расход",
        "списание",
        "chiqim",
        "debet",
    ),
    "credit": (
        "credit",
        "deposit",
        "deposits",
        "money in",
        "paid in",
        "кредит",
        "приход",
        "зачисление",
        "kirim",
        "kredit",
    ),
    "currency": ("currency", "ccy", "валюта", "valyuta", "val"),
    "balance": ("balance", "остаток", "баланс", "qoldiq"),
}

# известные «профили» выписок (для красивого имени формата в отчёте)
KNOWN_FORMATS: dict[str, tuple[str, ...]] = {
    "uz_debit_credit": ("date", "description", "debit", "credit", "balance"),
    "ru_signed_amount": ("дата", "операция", "сумма", "валюта"),
    "generic_en": ("posted date", "description", "amount", "currency", "reference"),
}

# порядок важен: при неоднозначности (все дни <= 12) побеждает «день первым»
DATE_FORMATS: tuple[str, ...] = (
    "%d.%m.%Y",
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%d.%m.%y",
    "%d/%m/%y",
    "%m/%d/%y",
    "%d %b %Y",
    "%b %d, %Y",
)


def norm_header(h: object) -> str:
    s = str(h).strip().lower().replace("﻿", "")
    s = re.sub(r"[_\s]+", " ", s)
    return s.strip(" .:")


class MappingError(ValueError):
    """Не удалось сопоставить колонки файла с полями транзакции."""


def detect_mapping(headers: Sequence[object]) -> dict[str, str]:
    """Сопоставить заголовки файла логическим полям (date/description/amount/...).

    Возвращает {логическое_поле: исходный_заголовок}. Требуется date, description и
    либо amount, либо пара debit/credit (хотя бы одна из них).
    """
    mapping: dict[str, str] = {}
    normalized = [(norm_header(h), str(h)) for h in headers]
    for field, synonyms in COLUMN_SYNONYMS.items():
        # сначала точное совпадение, затем «заголовок начинается с синонима»
        for nh, orig in normalized:
            if nh in synonyms and orig not in mapping.values():
                mapping[field] = orig
                break
        else:
            for nh, orig in normalized:
                if orig in mapping.values():
                    continue
                if any(nh.startswith(s + " ") or nh.startswith(s + "(") for s in synonyms):
                    mapping[field] = orig
                    break
    has_amount = "amount" in mapping or "debit" in mapping or "credit" in mapping
    if "date" not in mapping or "description" not in mapping or not has_amount:
        raise MappingError(
            f"Не удалось распознать колонки: {list(map(str, headers))}. "
            "Нужны дата, описание и сумма (или дебет/кредит)."
        )
    return mapping


def format_name_for(headers: Sequence[object]) -> str:
    key = tuple(norm_header(h) for h in headers if str(h).strip())
    for name, cols in KNOWN_FORMATS.items():
        if key == cols:
            return name
    return "auto"


def _date_part(value: str) -> str:
    s = str(value).strip()
    s = s.split("T")[0]
    # «05.04.2026 14:32», «2026-04-05 14:32:10» — время отбрасываем
    parts = s.split()
    if len(parts) > 1 and re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", parts[-1]):
        s = " ".join(parts[:-1])
    return s


def detect_date_format(values: Iterable[str]) -> str:
    """Подобрать формат даты, под который подходят ВСЕ непустые значения колонки."""
    sample = [_date_part(v) for v in values if str(v).strip()]
    if not sample:
        raise ValueError("Колонка дат пуста")
    for fmt in DATE_FORMATS:
        try:
            for v in sample:
                dt.datetime.strptime(v, fmt)
        except ValueError:
            continue
        return fmt
    raise ValueError(f"Неизвестный формат даты, пример: {sample[0]!r}")


def parse_date(value: str, fmt: str) -> dt.date:
    return dt.datetime.strptime(_date_part(value), fmt).date()


def frame_to_transactions(
    df: pd.DataFrame,
    mapping: dict[str, str],
    *,
    source: str,
    default_currency: str = "UZS",
    date_format: str | None = None,
) -> tuple[list[Transaction], str, int]:
    """Превратить таблицу (все значения — строки) в список транзакций.

    Возвращает (транзакции, формат_даты, число_пропущенных_строк).
    Пропускаются строки без даты/суммы (итоги, «входящий остаток» и т.п.).
    """
    date_col = mapping["date"]
    raw_dates = [str(v) for v in df[date_col].tolist()]
    candidate = [v for v in raw_dates if _looks_like_date(v)]
    fmt = date_format or detect_date_format(candidate)

    txns: list[Transaction] = []
    skipped = 0
    for _, row in df.iterrows():
        raw_date = str(row[date_col]).strip()
        if not _looks_like_date(raw_date):
            skipped += 1
            continue
        try:
            date = parse_date(raw_date, fmt)
        except ValueError:
            skipped += 1
            continue
        amount, direction = _row_amount(row, mapping)
        if amount is None:
            skipped += 1
            continue
        description = re.sub(r"\s+", " ", str(row[mapping["description"]])).strip()
        currency = (
            normalize_currency(row[mapping["currency"]], default_currency)
            if "currency" in mapping
            else default_currency
        )
        txns.append(
            Transaction(
                date=date,
                amount=round(abs(amount), 2),
                currency=currency,
                description=description,
                merchant_normalized=normalize_merchant(description),
                direction=direction,
                source=source,
            )
        )
    txns.sort(key=sort_key)
    return txns, fmt, skipped


def sort_key(t: Transaction) -> tuple:
    """Канонический порядок: независим от порядка строк в файле (новые сверху/снизу)."""
    return (t.date, t.direction, t.amount, t.description)


def _looks_like_date(value: str) -> bool:
    return bool(
        re.match(r"^\s*(\d{1,4}[./-]\d{1,2}[./-]\d{1,4}|\d{1,2} \w{3} \d{4}|\w{3} \d)", value)
    )


def _row_amount(row: pd.Series, mapping: dict[str, str]) -> tuple[float | None, str]:
    if "amount" in mapping:
        value = parse_amount(row[mapping["amount"]])
        if value is None:
            return None, "debit"
        return abs(value), ("debit" if value < 0 else "credit")
    debit = parse_amount(row[mapping["debit"]]) if "debit" in mapping else None
    credit = parse_amount(row[mapping["credit"]]) if "credit" in mapping else None
    if debit:
        return abs(debit), "debit"
    if credit:
        return abs(credit), "credit"
    return None, "debit"
