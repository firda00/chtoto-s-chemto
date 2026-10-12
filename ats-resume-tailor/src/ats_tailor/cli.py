"""CLI: ats score | tailor | cover-letter | vacancy."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .export import write_all, write_cover_letter
from .llm import LLM, get_llm
from .models import ATSScore, Vacancy
from .parsers import load_document, load_resume
from .scoring import score_resume
from .tailor import PLATFORM_RULES, cover_letter, tailor_resume
from .vacancy import extract_vacancy, is_real_llm

app = typer.Typer(
    help="ATS Resume Tailor: ATS Score резюме и адаптация под вакансию/платформу.",
    no_args_is_help=True,
    add_completion=False,
)

for _stream in (sys.stdout, sys.stderr):  # кириллица в консоли Windows
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

console = Console()

ResumeOpt = typer.Option(
    ..., "--resume", "-r", exists=True, dir_okay=False, help="Резюме: .pdf/.docx/.md/.txt"
)
VacancyOpt = typer.Option(
    ..., "--vacancy", "-v", exists=True, dir_okay=False, help="Текст вакансии"
)
MockOpt = typer.Option(
    None, "--mock/--llm", help="Принудительно MockLLM или реальная LLM (по умолчанию — авто по env)"
)
PlatformOpt = typer.Option("generic", "--platform", "-p", help="linkedin | indeed | hh | generic")


def _llm(mock: bool | None) -> LLM:
    llm = get_llm(mock=mock)
    console.print(f"[dim]LLM: {llm.name}[/dim]")
    return llm


def _vacancy(path: Path, llm: LLM | None) -> Vacancy:
    text = load_document(path).text
    return extract_vacancy(text, llm if is_real_llm(llm) else None)


def _check_platform(platform: str) -> str:
    if platform not in PLATFORM_RULES:
        raise typer.BadParameter(f"платформа должна быть одной из: {', '.join(PLATFORM_RULES)}")
    return platform


def print_score(score: ATSScore, vacancy: Vacancy) -> None:
    color = "green" if score.total >= 75 else "yellow" if score.total >= 50 else "red"
    console.print(f"\n[bold]Вакансия:[/bold] {vacancy.title}")
    console.print(f"[bold]ATS Score:[/bold] [{color}]{score.total:.1f} / 100[/{color}]\n")
    table = Table(title="Компоненты")
    table.add_column("Компонент")
    table.add_column("Вес", justify="right")
    table.add_column("Балл 0..1", justify="right")
    table.add_column("Очки", justify="right")
    table.add_column("Детали")
    for c in score.components:
        table.add_row(c.name, f"{c.weight:g}", f"{c.score:.2f}", f"{c.points:.1f}", c.details)
    console.print(table)
    if score.matched_keywords:
        console.print("[green]Найдены:[/green] " + ", ".join(score.matched_keywords))
    if score.missing_must_have:
        console.print("[red]Нет обязательных:[/red] " + ", ".join(score.missing_must_have))
    if score.missing_nice_to_have:
        console.print("[yellow]Нет желательных:[/yellow] " + ", ".join(score.missing_nice_to_have))
    if score.red_flags:
        console.print("[red]Red flags:[/red] " + "; ".join(score.red_flags))
    if score.suggestions:
        console.print("\n[bold]Рекомендации:[/bold]")
        for i, s in enumerate(score.suggestions, 1):
            console.print(f"  {i}. {s}")


@app.command()
def score(
    resume: Path = ResumeOpt,
    vacancy: Path = VacancyOpt,
    as_json: bool = typer.Option(False, "--json", help="Вывести JSON"),
    mock: bool | None = MockOpt,
) -> None:
    """Посчитать ATS Score резюме относительно вакансии."""
    llm = get_llm(mock=mock)
    vac = _vacancy(vacancy, llm)
    result = score_resume(load_resume(resume), vac)
    if as_json:
        typer.echo(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    else:
        print_score(result, vac)


@app.command()
def vacancy(
    vacancy: Path = VacancyOpt,
    mock: bool | None = MockOpt,
) -> None:
    """Показать, как распознана вакансия (JSON)."""
    llm = get_llm(mock=mock)
    typer.echo(json.dumps(_vacancy(vacancy, llm).model_dump(), ensure_ascii=False, indent=2))


@app.command()
def tailor(
    resume: Path = ResumeOpt,
    vacancy: Path = VacancyOpt,
    platform: str = PlatformOpt,
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Папка для результатов"),
    mock: bool | None = MockOpt,
) -> None:
    """Адаптировать резюме под вакансию и платформу; сохранить .md, .docx и changelog."""
    _check_platform(platform)
    llm = _llm(mock)
    vac = _vacancy(vacancy, llm)
    res = load_resume(resume)
    tr = tailor_resume(res, vac, llm, platform=platform)
    paths = write_all(tr, out, stem=resume.stem)
    console.print(f"\n[bold]ATS Score:[/bold] {tr.score_before} → [green]{tr.score_after}[/green]")
    table = Table(title="Changelog")
    table.add_column("Раздел")
    table.add_column("Действие")
    table.add_column("Описание")
    for ch in tr.changelog:
        table.add_row(ch.section, ch.action, ch.detail)
    console.print(table)
    for w in tr.warnings:
        console.print(f"[yellow]⚠ {w}[/yellow]")
    for p in paths:
        console.print(f"[dim]→ {p}[/dim]")


@app.command("cover-letter")
def cover_letter_cmd(
    resume: Path = ResumeOpt,
    vacancy: Path = VacancyOpt,
    platform: str = PlatformOpt,
    out: Path | None = typer.Option(None, "--out", "-o", help="Файл .md для письма"),
    mock: bool | None = MockOpt,
) -> None:
    """Сгенерировать сопроводительное письмо (только по фактам из резюме)."""
    _check_platform(platform)
    llm = _llm(mock)
    vac = _vacancy(vacancy, llm)
    cl = cover_letter(load_resume(resume), vac, llm, platform=platform)
    typer.echo(cl.text)
    for w in cl.warnings:
        console.print(f"[yellow]⚠ {w}[/yellow]")
    if out:
        target = out if out.suffix else out / f"cover_letter_{platform}.md"
        console.print(f"[dim]→ {write_cover_letter(cl, target)}[/dim]")


if __name__ == "__main__":  # pragma: no cover
    app()
