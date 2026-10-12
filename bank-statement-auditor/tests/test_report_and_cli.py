from __future__ import annotations

import json

from rich.console import Console
from typer.testing import CliRunner

from bank_auditor.cli import app
from bank_auditor.llm import MockLLM
from bank_auditor.pipeline import run_audit
from bank_auditor.report import print_report, render_html, render_markdown, write_reports

runner = CliRunner()


def test_markdown_report_sections(sample_report):
    md = render_markdown(sample_report)
    for section in (
        "## Траты по категориям",
        "## Топ мерчантов",
        "## Подписки и регулярные списания",
        "## Аномалии",
    ):
        assert section in md
    assert "ATLETIK FITNESS CLUB" in md and "ЗАБЫТА?" in md
    assert "29 900 → 39 900" in md
    assert "TEXNOMART SERGELI" in md
    assert "Еда и продукты" in md


def test_html_report_is_self_contained_and_escaped(sample_report):
    html = render_html(sample_report)
    assert html.startswith("<!doctype html>")
    assert "<script" not in html
    assert "ЗАБЫТА?" in html and "всплеск" in html
    # автоэкранирование: апостроф в имени мерчанта не ломает разметку
    assert "HUDUDGAZ TO&#39;LOV" in html


def test_write_reports(tmp_path, sample_report):
    paths = write_reports(sample_report, tmp_path / "out")
    assert {p.name for p in paths.values()} == {"report.md", "report.html", "report.json"}
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["n_transactions"] == sample_report.n_transactions
    assert "transactions" not in data  # сырые операции в JSON не выгружаются


def test_console_report(sample_report):
    console = Console(record=True, width=160)
    print_report(sample_report, console)
    text = console.export_text()
    assert "Регулярные списания" in text and "Аномалии" in text and "ЗАБЫТА?" in text


def test_run_audit_on_all_formats_gives_same_numbers(data_dir):
    reports = [
        run_audit(data_dir / n)
        for n in ("statement_uzbank.csv", "statement_rubank.csv", "statement_sample.csv")
    ]
    sigs = {
        (r.total_spent, r.monthly_burn, tuple(s.merchant for s in r.subscriptions)) for r in reports
    }
    assert len(sigs) == 1


def test_run_audit_with_mock_llm(data_dir):
    llm = MockLLM(rules={"ZIYO NUR": {"category": "shopping", "confidence": 0.9}})
    r = run_audit(data_dir / "statement_sample.csv", llm=llm)
    assert any(
        t.merchant_normalized == "ZIYO NUR" and t.category == "shopping" for t in r.transactions
    )


def test_cli_analyze_smoke(data_dir, tmp_path):
    out = tmp_path / "out"
    res = runner.invoke(app, ["analyze", str(data_dir / "statement_sample.csv"), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert (out / "report.md").exists() and (out / "report.html").exists()
    assert "ATLETIK FITNESS CLUB" in res.output


def test_cli_subscriptions_smoke(data_dir):
    res = runner.invoke(app, ["subscriptions", str(data_dir / "statement_uzbank.csv")])
    assert res.exit_code == 0, res.output
    assert "NETFLIX" in res.output and "Похоже на забытые" in res.output


def test_cli_categorize_smoke_with_llm_flag(data_dir, tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    csv_out = tmp_path / "cat.csv"
    res = runner.invoke(
        app,
        ["categorize", str(data_dir / "statement_small.pdf"), "--llm", "--csv", str(csv_out)],
    )
    assert res.exit_code == 0, res.output
    assert "MockLLM" in res.output
    assert csv_out.exists()


def test_cli_missing_file():
    res = runner.invoke(app, ["analyze", "no_such_file.csv"])
    assert res.exit_code != 0
