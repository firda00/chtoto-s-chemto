"""CLI: `audit analyze | subscriptions | categorize | formats`."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .categorize import Categorizer
from .llm import LLM, MockLLM, get_llm
from .models import CATEGORY_LABELS_RU
from .parsers import load_statement
from .parsers.common import KNOWN_FORMATS
from .pipeline import build_report
from .report import money, print_report, subscriptions_table, write_reports
from .subscriptions import burn, detect_subscriptions

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Аудит банковских выписок: категории трат, подписки, забытые списания.",
)

for _stream in (sys.stdout, sys.stderr):  # Windows-консоль с cp1251/cp866 не должна падать
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

FileArg = typer.Argument(..., exists=True, dir_okay=False, help="Выписка: CSV или PDF")
CurrencyOpt = typer.Option("UZS", "--currency", help="Валюта, если в файле её нет")
LlmOpt = typer.Option(
    False, "--llm", help="LLM-фолбэк для неизвестных мерчантов (нужен LLM_API_KEY)"
)


def _console() -> Console:
    return Console(width=140 if not sys.stdout.isatty() else None)


def _llm(enabled: bool, console: Console) -> LLM | None:
    if not enabled:
        return None
    llm = get_llm()
    if isinstance(llm, MockLLM):
        console.print("[yellow]LLM_API_KEY не задан — используется MockLLM (без сети).[/yellow]")
    return llm


def _load(path: Path, currency: str, llm: LLM | None, console: Console):
    parsed = load_statement(path, default_currency=currency)
    console.print(
        f"[dim]Формат: {parsed.format_name}; кодировка: {parsed.encoding or '—'}; "
        f"разделитель: {parsed.delimiter!r}; даты: {parsed.date_format}; "
        f"операций: {len(parsed.transactions)}; пропущено строк: {parsed.skipped_rows}[/dim]"
    )
    categorizer = Categorizer(llm=llm)
    txns = categorizer.categorize(parsed.transactions)
    if llm is not None:
        console.print(
            f"[dim]LLM-запросов: {categorizer.llm_calls}, ошибок: {categorizer.llm_errors}[/dim]"
        )
    return parsed, txns


@app.command()
def analyze(
    file: Path = FileArg,
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Папка для report.md/html/json"),
    currency: str = CurrencyOpt,
    llm: bool = LlmOpt,
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Не печатать отчёт в консоль"),
) -> None:
    """Полный аудит: категории, топ мерчантов, подписки, аномалии + отчёты в --out."""
    console = _console()
    parsed, txns = _load(file, currency, _llm(llm, console), console)
    report = build_report(txns, source_file=file.name, format_name=parsed.format_name)
    if not quiet:
        print_report(report, console)
    paths = write_reports(report, out)
    console.print("Отчёты: " + ", ".join(str(p) for p in paths.values()))


@app.command()
def subscriptions(
    file: Path = FileArg,
    currency: str = CurrencyOpt,
    llm: bool = LlmOpt,
) -> None:
    """Только регулярные списания и «забытые» подписки."""
    console = _console()
    _, txns = _load(file, currency, _llm(llm, console), console)
    subs = detect_subscriptions(txns)
    if not subs:
        console.print("Регулярных списаний не найдено.")
        return
    console.print(subscriptions_table(subs))
    monthly, annual = burn(subs)
    cur = subs[0].currency
    forgotten = [s for s in subs if s.forgotten]
    console.print(
        f"Нагрузка подписок: [bold]{money(monthly, cur)}/мес[/bold], {money(annual, cur)}/год"
    )
    if forgotten:
        saving = sum(s.annual_cost for s in forgotten)
        names = ", ".join(s.merchant for s in forgotten)
        console.print(
            f"[bold red]Похоже на забытые:[/bold red] {names} — экономия {money(saving, cur)}/год"
        )


@app.command()
def categorize(
    file: Path = FileArg,
    currency: str = CurrencyOpt,
    llm: bool = LlmOpt,
    limit: int = typer.Option(30, "--limit", "-n", help="Сколько операций показать (0 — все)"),
    csv_out: Path | None = typer.Option(None, "--csv", help="Сохранить размеченные операции в CSV"),
) -> None:
    """Разметить операции по категориям и показать сводку."""
    console = _console()
    _, txns = _load(file, currency, _llm(llm, console), console)
    table = Table(title="Операции", header_style="bold")
    for col in ("Дата", "Мерчант", "Категория", "Сумма"):
        table.add_column(col, justify="right" if col == "Сумма" else "left")
    shown = txns if limit == 0 else txns[:limit]
    for t in shown:
        sign = "-" if t.direction == "debit" else "+"
        table.add_row(
            f"{t.date:%d.%m.%Y}",
            t.merchant_normalized,
            CATEGORY_LABELS_RU.get(t.category or "other", t.category or ""),
            f"{sign}{money(t.amount, t.currency)}",
        )
    console.print(table)
    counts = Counter(t.category for t in txns)
    summary = ", ".join(
        f"{CATEGORY_LABELS_RU.get(k or 'other', k)}: {v}" for k, v in counts.most_common()
    )
    console.print(f"Всего {len(txns)} операций. {summary}")
    if csv_out is not None:
        import pandas as pd

        csv_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([t.model_dump() for t in txns]).to_csv(csv_out, index=False, encoding="utf-8")
        console.print(f"Сохранено: {csv_out}")


@app.command()
def formats() -> None:
    """Показать поддерживаемые форматы выписок."""
    console = _console()
    for name, cols in KNOWN_FORMATS.items():
        console.print(f"[bold]{name}[/bold]: {', '.join(cols)}")
    console.print("Плюс автоопределение колонок по синонимам (RU/EN/UZ) и PDF (таблицы/строки).")


if __name__ == "__main__":  # pragma: no cover
    app()
