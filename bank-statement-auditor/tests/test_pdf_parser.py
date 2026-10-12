from __future__ import annotations

import datetime as dt

from bank_auditor.parsers import load_statement, parse_csv, parse_pdf, parse_text_lines, pick_parser
from bank_auditor.parsers.pdf_parser import parse_pdf as pdf_fn


def _key(t):
    return tuple(t.model_dump(exclude={"source"}).items())


def test_pdf_table_parse_matches_csv(data_dir):
    res = parse_pdf(data_dir / "statement_small.pdf")
    assert res.format_name.startswith("pdf/table")
    assert len(res.transactions) >= 30
    csv_rows = {_key(t) for t in parse_csv(data_dir / "statement_sample.csv").transactions}
    for t in res.transactions:
        assert t.date.month == 4
        assert _key(t) in csv_rows


def test_registry_picks_parser_by_extension_and_magic(data_dir, tmp_path):
    assert pick_parser(data_dir / "statement_small.pdf") is pdf_fn
    renamed = tmp_path / "statement.bin"
    renamed.write_bytes((data_dir / "statement_small.pdf").read_bytes())
    assert pick_parser(renamed) is pdf_fn
    assert len(load_statement(renamed).transactions) == len(
        load_statement(data_dir / "statement_small.pdf").transactions
    )


def test_line_regex_fallback_on_text():
    text = (
        "Namuna Bank\n"
        "Statement for April 2026\n"
        "05.04.2026 NETFLIX.COM 4169****7788 AMSTERDAM -126 300.00 UZS\n"
        "08.04.2026 SPOTIFY STOCKHOLM -75 000,00\n"
        "25.04.2026 ZARPLATA DIGITAL SOLUTIONS 12 500 000.00 UZS\n"
        "Итого: 3 операции\n"
    )
    res = parse_text_lines(text)
    assert res.format_name == "pdf/lines"
    assert [(t.date, t.amount, t.direction) for t in res.transactions] == [
        (dt.date(2026, 4, 5), 126300.0, "debit"),
        (dt.date(2026, 4, 8), 75000.0, "debit"),
        (dt.date(2026, 4, 25), 12_500_000.0, "credit"),
    ]
    assert res.transactions[0].merchant_normalized == "NETFLIX"


def test_pdf_without_table_uses_line_fallback(tmp_path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    path = tmp_path / "plain.pdf"
    c = canvas.Canvas(str(path), pagesize=A4)
    lines = [
        "Plain text statement",
        "03.05.2026  ATLETIK FITNESS CLUB TOSHKENT  -350 000.00  UZS",
        "05.05.2026  NETFLIX.COM AMSTERDAM  -126 000.00  UZS",
        "25.05.2026  ZARPLATA DIGITAL SOLUTIONS  12 500 000.00  UZS",
    ]
    for i, line in enumerate(lines):
        c.drawString(50, 800 - 16 * i, line)
    c.save()
    res = parse_pdf(path)
    assert res.format_name == "pdf/lines"
    assert [t.merchant_normalized for t in res.transactions] == [
        "ATLETIK FITNESS CLUB",
        "NETFLIX",
        "ZARPLATA DIGITAL SOLUTIONS",
    ]
    assert res.transactions[-1].direction == "credit"
