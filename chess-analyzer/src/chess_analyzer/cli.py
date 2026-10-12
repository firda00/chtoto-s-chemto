"""CLI: ``chess analyze <pgn>`` и ``chess summary <pgn>``."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.table import Table

from .analysis import CLASS_ORDER, GameAnalysis, MoveClass, analyse_game
from .engine import make_engine
from .explain import CLASS_NAMES, explain_game
from .llm import get_llm
from .pgn import PGNError, count_plies, game_headers, load_games
from .report import write_report

app = typer.Typer(add_completion=False, help="Chess Match Analyzer & Mistake Explainer")
console = Console()

CLASS_STYLE = {
    MoveClass.INACCURACY: "yellow",
    MoveClass.MISTAKE: "dark_orange",
    MoveClass.BLUNDER: "bold red",
}


def _load(pgn: Path):
    try:
        return load_games(pgn)
    except (PGNError, OSError) as exc:
        console.print(f"[red]Ошибка чтения PGN:[/red] {exc}")
        raise typer.Exit(code=2) from exc


def _analyse_all(games, engine_kind: str, depth: int, fake_depth: int) -> list[GameAnalysis]:
    engine = make_engine(engine_kind, fake_depth=fake_depth)
    results = []
    try:
        with Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
            transient=True,
        ) as progress:
            for i, game in enumerate(games, start=1):
                task = progress.add_task(f"Партия {i}/{len(games)}", total=count_plies(game) + 1)
                results.append(
                    analyse_game(
                        game,
                        engine,
                        depth=depth,
                        progress=lambda done, _total, t=task: progress.update(t, completed=done),
                    )
                )
    finally:
        engine.close()
    return results


def _print_summary(analyses: list[GameAnalysis], lang: str) -> None:
    names = CLASS_NAMES["ru" if lang.startswith("ru") else "en"]
    for ga in analyses:
        table = Table(title=f"{ga.title}  [{ga.result}]  ({ga.engine})")
        table.add_column("Игрок")
        table.add_column("Точность", justify="right")
        table.add_column("ACPL", justify="right")
        for c in (MoveClass.INACCURACY, MoveClass.MISTAKE, MoveClass.BLUNDER):
            table.add_column(names[c], justify="right", style=CLASS_STYLE[c])
        for p in (ga.white, ga.black):
            table.add_row(
                f"{p.name} ({p.color})",
                f"{p.accuracy:.1f}%" if p.accuracy is not None else "—",
                f"{p.acpl:.1f}" if p.acpl is not None else "—",
                *(str(p.counts.get(c.value, 0)) for c in CLASS_ORDER[3:]),
            )
        console.print(table)
        for m in ga.critical_moves(MoveClass.MISTAKE):
            style = CLASS_STYLE.get(m.classification, "")
            best = f" (лучше {m.best_move_san})" if m.best_move_san else ""
            console.print(
                f"  [{style}]{m.label} — {names[m.classification]}[/]{best}, -{m.cp_loss} cp"
            )


@app.command()
def analyze(
    pgn: Path = typer.Argument(..., exists=True, dir_okay=False, help="PGN-файл (1+ партий)"),
    engine: str = typer.Option("auto", "--engine", "-e", help="fake | stockfish | api | auto"),
    depth: int = typer.Option(14, "--depth", "-d", help="Глубина анализа"),
    fake_depth: int = typer.Option(
        2, "--fake-depth", help="Глубина перебора FakeEngine (2 или 4; нечётная округляется вниз)"
    ),
    lang: str = typer.Option("ru", "--lang", "-l", help="Язык объяснений: ru | en"),
    llm: str = typer.Option("auto", "--llm", help="openai | mock | none | auto"),
    min_class: MoveClass = typer.Option(
        MoveClass.MISTAKE, "--min-class", help="С какого класса объяснять ходы"
    ),
    out: Path = typer.Option(Path("out"), "--out", "-o", help="Папка для отчёта"),
) -> None:
    """Проанализировать партии, объяснить ошибки и записать отчёт (MD + HTML + JSON + SVG)."""
    games = _load(pgn)
    console.print(f"Загружено партий: [bold]{len(games)}[/bold] из {pgn}")
    analyses = _analyse_all(games, engine, depth, fake_depth)
    llm_client = get_llm(llm)
    console.print(f"Объяснения: [bold]{llm_client.name if llm_client else 'шаблон'}[/bold]")
    for ga in analyses:
        explain_game(ga, llm_client, lang=lang, min_class=min_class)
    paths = write_report(analyses, out, lang=lang, min_class=min_class)
    _print_summary(analyses, lang)
    for kind, path in paths.items():
        console.print(f"[green]{kind}[/green]: {path}")


@app.command()
def summary(
    pgn: Path = typer.Argument(..., exists=True, dir_okay=False, help="PGN-файл"),
    engine: str | None = typer.Option(
        None, "--engine", "-e", help="Если указан — посчитать точность (fake|stockfish|api|auto)"
    ),
    depth: int = typer.Option(12, "--depth", "-d"),
    fake_depth: int = typer.Option(2, "--fake-depth"),
    lang: str = typer.Option("ru", "--lang", "-l"),
) -> None:
    """Краткая сводка по партиям: заголовки и число ходов (+ точность, если задан движок)."""
    games = _load(pgn)
    table = Table(title=f"{pgn.name}: партий — {len(games)}")
    for col in ("#", "Белые", "Чёрные", "Результат", "Событие", "Дата", "ECO", "Полуходов"):
        table.add_column(col)
    for i, game in enumerate(games, start=1):
        h = game_headers(game)
        table.add_row(
            str(i),
            h.get("White", "?"),
            h.get("Black", "?"),
            h.get("Result", "*"),
            h.get("Event", "?"),
            h.get("Date", "?"),
            h.get("ECO", "—"),
            str(count_plies(game)),
        )
    console.print(table)
    if engine:
        _print_summary(_analyse_all(games, engine, depth, fake_depth), lang)


if __name__ == "__main__":  # pragma: no cover
    app()
