from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

from bank_auditor.parsers import load_statement, parse_csv, sniff_delimiter, sniff_encoding
from bank_auditor.parsers.common import MappingError, detect_date_format, detect_mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_statements  # noqa: E402

FILES = {
    "statement_uzbank.csv": ("csv/uz_debit_credit", "cp1251", ";", "%d.%m.%Y"),
    "statement_rubank.csv": ("csv/ru_signed_amount", "utf-8", ",", "%Y-%m-%d"),
    "statement_sample.csv": ("csv/generic_en", "utf-8", ",", "%m/%d/%Y"),
}


def _comparable(result):
    return [t.model_dump(exclude={"source"}) for t in result.transactions]


@pytest.mark.parametrize("name", list(FILES))
def test_format_autodetection(data_dir, name):
    fmt, enc, delim, date_fmt = FILES[name]
    res = parse_csv(data_dir / name)
    assert res.format_name == fmt
    assert res.encoding == enc
    assert res.delimiter == delim
    assert res.date_format == date_fmt
    assert 280 <= len(res.transactions) <= 340


def test_all_three_formats_parse_to_identical_transactions(data_dir):
    results = [_comparable(parse_csv(data_dir / n)) for n in FILES]
    assert results[0] == results[1] == results[2]
    # и это действительно те операции, что генерировал скрипт
    events = make_statements.build_events()
    assert len(results[0]) == len(events)
    credits = sum(1 for t in results[0] if t["direction"] == "credit")
    assert credits == sum(1 for e in events if e.amount > 0)


def test_uzbank_skips_preamble_and_totals(data_dir):
    res = parse_csv(data_dir / "statement_uzbank.csv")
    assert res.skipped_rows == 1  # строка «Итого»
    assert res.column_mapping["debit"] == "debit"
    assert all(t.currency == "UZS" for t in res.transactions)


def test_generator_is_deterministic(tmp_path, data_dir):
    events = make_statements.build_events()
    make_statements.write_generic(events, tmp_path / "s.csv")
    make_statements.write_uzbank(events, tmp_path / "u.csv")
    assert (tmp_path / "s.csv").read_bytes() == (data_dir / "statement_sample.csv").read_bytes()
    assert (tmp_path / "u.csv").read_bytes() == (data_dir / "statement_uzbank.csv").read_bytes()


def test_cp1251_encoding_sniffed(tmp_path):
    content = (
        "Дата;Описание;Сумма;Валюта\n"
        "05.04.2026;Покупка КОРЗИНКА Ташкент;-150 000,00;сум\n"
        "06.04.2026;ЗАРПЛАТА ООО Ромашка;5 000 000,00;сум\n"
    )
    path = tmp_path / "cp.csv"
    path.write_bytes(content.encode("cp1251"))
    assert sniff_encoding(path.read_bytes()) == "cp1251"
    res = parse_csv(path)
    assert res.encoding == "cp1251"
    assert [t.description for t in res.transactions] == [
        "Покупка КОРЗИНКА Ташкент",
        "ЗАРПЛАТА ООО Ромашка",
    ]
    first, second = res.transactions
    assert (first.amount, first.direction, first.currency) == (150000.0, "debit", "UZS")
    assert (second.amount, second.direction) == (5_000_000.0, "credit")
    assert first.merchant_normalized == "KORZINKA"


def test_utf8_bom_and_tab_delimiter(tmp_path):
    path = tmp_path / "bom.tsv"
    path.write_bytes(
        "﻿Transaction Date\tDetails\tWithdrawal\tDeposit\n"
        "2026-04-01\tNETFLIX.COM\t9.99\t\n2026-04-02\tSALARY\t\t1500.00\n".encode()
    )
    res = load_statement(path, default_currency="USD")
    assert res.encoding == "utf-8-sig"
    assert res.delimiter == "\t"
    assert [(t.amount, t.direction, t.currency) for t in res.transactions] == [
        (9.99, "debit", "USD"),
        (1500.0, "credit", "USD"),
    ]


def test_sniff_delimiter():
    assert sniff_delimiter("a;b;c\n1;2;3\n") == ";"
    assert sniff_delimiter("a,b,c\n1,2,3\n") == ","
    assert sniff_delimiter("a|b|c\n1|2|3\n") == "|"


def test_date_format_detection_day_vs_month_first():
    assert detect_date_format(["01/02/2026", "25/02/2026"]) == "%d/%m/%Y"
    assert detect_date_format(["01/02/2026", "02/25/2026"]) == "%m/%d/%Y"
    assert detect_date_format(["2026-04-05 14:32", "2026-04-06T10:00:00"]) == "%Y-%m-%d"
    assert detect_date_format(["05.04.26"]) == "%d.%m.%y"
    with pytest.raises(ValueError):
        detect_date_format(["вчера"])


def test_mapping_synonyms_and_errors():
    m = detect_mapping(["Sana", "Tavsif", "Chiqim", "Kirim", "Qoldiq"])
    assert m == {
        "date": "Sana",
        "description": "Tavsif",
        "debit": "Chiqim",
        "credit": "Kirim",
        "balance": "Qoldiq",
    }
    with pytest.raises(MappingError):
        detect_mapping(["foo", "bar", "baz"])


def test_parsed_dates_and_sorting(data_dir):
    txns = parse_csv(data_dir / "statement_sample.csv").transactions
    assert txns[0].date == dt.date(2026, 4, 1)
    assert txns[-1].date == dt.date(2026, 9, 30)
    assert [t.date for t in txns] == sorted(t.date for t in txns)
