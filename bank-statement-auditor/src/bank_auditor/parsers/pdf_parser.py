"""PDF-парсер: таблицы через pdfplumber, запасной вариант — построчный regex по тексту."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pdfplumber

from ..models import ParseResult, Transaction
from ..normalize import normalize_currency, normalize_merchant, parse_amount
from .common import (
    MappingError,
    detect_date_format,
    detect_mapping,
    format_name_for,
    frame_to_transactions,
    parse_date,
    sort_key,
)

# «05.04.2026  NETFLIX.COM AMSTERDAM  -126 300.00  UZS» (валюта и знак — опциональны)
LINE_RE = re.compile(
    r"^\s*(?P<date>\d{1,4}[./-]\d{1,2}[./-]\d{2,4})\s+"
    r"(?P<desc>.+?)\s+"
    r"(?P<amount>[-+−]?\(?\d{1,3}(?:[  ,.]?\d{3})*(?:[.,]\d{1,2})?\)?)"
    r"(?:\s+(?P<currency>[A-Za-z]{3}|сум|руб\.?))?\s*$"
)


def parse_pdf(path: str | Path, *, default_currency: str = "UZS") -> ParseResult:
    path = Path(path)
    with pdfplumber.open(path) as pdf:
        result = _parse_tables(pdf, path, default_currency)
        if result is not None and result.transactions:
            return result
        text = "\n".join((page.extract_text() or "") for page in pdf.pages)
    return parse_text_lines(
        text, source=f"pdf:lines:{path.name}", default_currency=default_currency
    )


def _parse_tables(pdf: pdfplumber.PDF, path: Path, default_currency: str) -> ParseResult | None:
    header: list[str] | None = None
    rows: list[list[str]] = []
    for page in pdf.pages:
        for table in page.extract_tables():
            for raw_row in table:
                cells = [(c or "").replace("\n", " ").strip() for c in raw_row]
                if not any(cells):
                    continue
                if header is None:
                    try:
                        detect_mapping(cells)
                    except MappingError:
                        continue
                    header = cells
                    continue
                if cells == header:  # заголовок, повторённый на следующей странице
                    continue
                if len(cells) == len(header):
                    rows.append(cells)
    if header is None or not rows:
        return None
    df = pd.DataFrame(rows, columns=header)
    mapping = detect_mapping(header)
    fmt_name = format_name_for(header)
    txns, date_fmt, skipped = frame_to_transactions(
        df, mapping, source=f"pdf:table:{path.name}", default_currency=default_currency
    )
    return ParseResult(
        transactions=txns,
        format_name=f"pdf/table/{fmt_name}",
        date_format=date_fmt,
        column_mapping=mapping,
        skipped_rows=skipped,
    )


def parse_text_lines(
    text: str, *, source: str = "pdf:lines", default_currency: str = "UZS"
) -> ParseResult:
    """Fallback: выписка без «настоящих» таблиц — ищем строки вида «дата описание сумма [вал]».

    Знак суммы: «-» → списание, без знака → зачисление (как в большинстве выписок).
    """
    matches = [m for line in text.splitlines() if (m := LINE_RE.match(line))]
    if not matches:
        return ParseResult(transactions=[], format_name="pdf/lines", skipped_rows=0)
    date_fmt = detect_date_format([m["date"] for m in matches])
    txns: list[Transaction] = []
    skipped = 0
    for m in matches:
        value = parse_amount(m["amount"])
        if value is None:
            skipped += 1
            continue
        desc = re.sub(r"\s+", " ", m["desc"]).strip()
        txns.append(
            Transaction(
                date=parse_date(m["date"], date_fmt),
                amount=round(abs(value), 2),
                currency=normalize_currency(m["currency"], default_currency),
                description=desc,
                merchant_normalized=normalize_merchant(desc),
                direction="debit" if value < 0 else "credit",
                source=source,
            )
        )
    txns.sort(key=sort_key)
    return ParseResult(
        transactions=txns, format_name="pdf/lines", date_format=date_fmt, skipped_rows=skipped
    )
