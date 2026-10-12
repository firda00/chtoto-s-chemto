"""Отчёты: консоль (rich), Markdown и HTML (jinja2)."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, PackageLoader
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .models import CATEGORY_LABELS_RU, AuditReport, Subscription

PERIOD_RU = {
    "weekly": "еженедельно",
    "biweekly": "раз в 2 недели",
    "monthly": "ежемесячно",
    "quarterly": "ежеквартально",
    "yearly": "ежегодно",
}


def money(x: float, currency: str = "") -> str:
    s = f"{x:,.0f}".replace(",", " ")
    return f"{s} {currency}".strip()


def cat_label(cat: str | None) -> str:
    return CATEGORY_LABELS_RU.get(cat or "other", cat or "—")


def sub_status(s: Subscription) -> str:
    if not s.active:
        return "неактивна"
    if s.forgotten:
        return "ЗАБЫТА?"
    if s.flags:
        return "проверить"
    return "ок"


def _env() -> Environment:
    env = Environment(
        loader=PackageLoader("bank_auditor", "templates"),
        autoescape=lambda name: bool(name) and ".html" in name,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["money"] = money
    env.filters["cat"] = cat_label
    env.filters["status"] = sub_status
    env.filters["period"] = lambda p: PERIOD_RU.get(p, p)
    return env


def _context(report: AuditReport) -> dict:
    total = report.total_spent or 1.0
    max_cat = max(report.spending_by_category.values(), default=1.0) or 1.0
    categories = [
        {
            "key": k,
            "label": cat_label(k),
            "amount": v,
            "share": v / total * 100,
            "bar": v / max_cat * 100,
        }
        for k, v in report.spending_by_category.items()
    ]
    return {
        "r": report,
        "categories": categories,
        "active_subs": [s for s in report.subscriptions if s.active],
        "inactive_subs": [s for s in report.subscriptions if not s.active],
        "forgotten": [s for s in report.subscriptions if s.forgotten],
    }


def render_markdown(report: AuditReport) -> str:
    return _env().get_template("report.md.j2").render(**_context(report))


def render_html(report: AuditReport) -> str:
    return _env().get_template("report.html.j2").render(**_context(report))


def write_reports(report: AuditReport, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "md": out / "report.md",
        "html": out / "report.html",
        "json": out / "report.json",
    }
    paths["md"].write_text(render_markdown(report), encoding="utf-8")
    paths["html"].write_text(render_html(report), encoding="utf-8")
    paths["json"].write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return paths


# --------------------------------------------------------------------------- консоль


def subscriptions_table(
    report_or_subs: AuditReport | list[Subscription], currency: str = ""
) -> Table:
    subs = (
        report_or_subs.subscriptions if isinstance(report_or_subs, AuditReport) else report_or_subs
    )
    table = Table(title="Регулярные списания", show_lines=False, header_style="bold")
    table.add_column("Мерчант")
    table.add_column("Сумма", justify="right")
    table.add_column("Период")
    table.add_column("Раз", justify="right")
    table.add_column("В год", justify="right")
    table.add_column("Увер.", justify="right")
    table.add_column("Статус")
    table.add_column("Отменить?")
    for s in subs:
        style = "bold red" if s.forgotten else ("dim" if not s.active else "")
        table.add_row(
            s.merchant,
            money(s.amount),
            f"{PERIOD_RU.get(s.period_label, s.period_label)} (~{s.period_days} дн.)",
            str(s.occurrences),
            money(s.annual_cost, currency or s.currency),
            f"{s.confidence:.2f}",
            sub_status(s),
            s.hint,
            style=style,
        )
    return table


def print_report(report: AuditReport, console: Console | None = None) -> None:
    console = console or Console()
    c = report.currency
    console.print(
        Panel.fit(
            f"[bold]Аудит выписки[/bold] {report.source_file} ({report.format_name})\n"
            f"Период: {report.period_start:%d.%m.%Y} — {report.period_end:%d.%m.%Y}, "
            f"операций: {report.n_transactions}\n"
            f"Потрачено: [bold]{money(report.total_spent, c)}[/bold]   "
            f"Поступления: {money(report.total_income, c)}\n"
            f"Подписки: [bold]{money(report.monthly_burn, c)}/мес[/bold] "
            f"({money(report.annual_burn, c)}/год); "
            f"забытые: [bold red]{money(report.forgotten_annual_savings, c)}/год[/bold red]",
            title="bank-auditor",
        )
    )

    cats = Table(title="Траты по категориям", header_style="bold")
    cats.add_column("Категория")
    cats.add_column("Сумма", justify="right")
    cats.add_column("Доля", justify="right")
    cats.add_column("")
    max_v = max(report.spending_by_category.values(), default=1) or 1
    for k, v in report.spending_by_category.items():
        share = v / (report.total_spent or 1) * 100
        cats.add_row(
            cat_label(k), money(v, c), f"{share:.1f}%", "#" * max(1, round(v / max_v * 24))
        )
    console.print(cats)

    top = Table(title="Топ мерчантов", header_style="bold")
    top.add_column("Мерчант")
    top.add_column("Категория")
    top.add_column("Сумма", justify="right")
    top.add_column("Операций", justify="right")
    for m in report.top_merchants:
        top.add_row(m.merchant, cat_label(m.category), money(m.total, c), str(m.count))
    console.print(top)

    console.print(subscriptions_table(report, c))
    for s in report.subscriptions:
        if s.reasons and s.active:
            console.print(f"  [bold]{s.merchant}[/bold]: " + "; ".join(s.reasons))

    if report.anomalies:
        an = Table(title="Аномалии", header_style="bold")
        an.add_column("Дата")
        an.add_column("Тип")
        an.add_column("Мерчант")
        an.add_column("Сумма", justify="right")
        an.add_column("Детали")
        for a in report.anomalies:
            kind = "дубль" if a.kind == "duplicate" else "всплеск"
            an.add_row(
                f"{a.date:%d.%m.%Y}", kind, a.merchant, money(a.amount, a.currency), a.details
            )
        console.print(an)
