"""CSV-парсер: определение кодировки, разделителя, колонок и формата дат."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import pandas as pd

from ..models import ParseResult
from .common import detect_mapping, format_name_for, frame_to_transactions

ENCODING_CANDIDATES = ("utf-8", "cp1251")


def sniff_encoding(raw: bytes) -> str:
    """utf-8 (с BOM или без) → cp1251 (типично для выгрузок из 1С/банк-клиентов) → latin-1."""
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    for enc in ENCODING_CANDIDATES:
        try:
            raw.decode(enc)
        except UnicodeDecodeError:
            continue
        return enc
    return "latin-1"


def sniff_delimiter(text: str) -> str:
    """csv.Sniffer по первым строкам; запасной вариант — самый частый из ; , \\t |."""
    head = "\n".join(text.splitlines()[:20])
    try:
        return csv.Sniffer().sniff(head, delimiters=";,\t|").delimiter
    except csv.Error:
        first = text.splitlines()[0] if text else ""
        counts = {d: first.count(d) for d in (";", ",", "\t", "|")}
        return max(counts, key=counts.get) if any(counts.values()) else ","


def _find_header_row(lines: list[str], delimiter: str) -> int:
    """Пропустить «шапку» выписки (название банка, номер счёта) до строки заголовков."""
    for i, line in enumerate(lines[:30]):
        cells = next(csv.reader([line], delimiter=delimiter), [])
        if len(cells) < 3:
            continue
        try:
            detect_mapping(cells)
        except ValueError:
            continue
        return i
    return 0


def parse_csv(
    path: str | Path,
    *,
    default_currency: str = "UZS",
    encoding: str | None = None,
    delimiter: str | None = None,
) -> ParseResult:
    """Разобрать CSV-выписку любого из поддерживаемых форматов (автоопределение)."""
    path = Path(path)
    raw = path.read_bytes()
    enc = encoding or sniff_encoding(raw)
    text = raw.decode(enc)
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    delim = delimiter or sniff_delimiter(text)
    lines = text.split("\n")
    header_idx = _find_header_row(lines, delim)
    body = "\n".join(lines[header_idx:])
    df = pd.read_csv(
        io.StringIO(body),
        sep=delim,
        dtype=str,
        keep_default_na=False,
        skip_blank_lines=True,
        engine="python",
    )
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    mapping = detect_mapping(list(df.columns))
    fmt_name = format_name_for(list(df.columns))
    source = f"csv:{fmt_name}:{path.name}"
    txns, date_fmt, skipped = frame_to_transactions(
        df, mapping, source=source, default_currency=default_currency
    )
    return ParseResult(
        transactions=txns,
        format_name=f"csv/{fmt_name}",
        encoding=enc,
        delimiter=delim,
        date_format=date_fmt,
        column_mapping=mapping,
        skipped_rows=skipped,
    )
