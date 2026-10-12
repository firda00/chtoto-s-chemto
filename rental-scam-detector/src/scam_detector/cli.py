"""CLI: uv run scam make-dataset | train | check <file.json> | batch <dir> | serve."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .config import DATA_DIR, PROJECT_ROOT
from .config import model_path as default_model_path
from .features import load_corpus, load_listing
from .models import RiskLevel, ScamReport

app = typer.Typer(
    help="Local Rental Listing Scam Detector — Scam Score объявлений аренды.", no_args_is_help=True
)
console = Console()

RISK_STYLE = {
    RiskLevel.LOW: "green",
    RiskLevel.MEDIUM: "yellow",
    RiskLevel.HIGH: "dark_orange",
    RiskLevel.CRITICAL: "bold red",
}


@app.callback()
def _main() -> None:
    # Windows-консоль: кириллица/узбекские апострофы без UnicodeEncodeError при перенаправлении
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


@app.command("make-dataset")
def make_dataset_cmd(
    n: int = typer.Option(2000, help="Число объявлений"),
    scam_ratio: float = typer.Option(0.25, help="Доля мошеннических объявлений"),
    seed: int = typer.Option(42),
    out: Path = typer.Option(DATA_DIR / "dataset.csv", help="Куда сохранить CSV"),
) -> None:
    """Синтетический размеченный датасет признаков."""
    from .synth import make_dataset

    df = make_dataset(n, scam_ratio, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    console.print(f"[green]OK[/] {out}: {len(df)} строк, доля scam = {df['label'].mean():.1%}")


@app.command()
def train(
    data: Path = typer.Option(DATA_DIR / "dataset.csv", help="CSV из make-dataset"),
    out_dir: Path = typer.Option(PROJECT_ROOT / "models", help="Папка для модели и метрик"),
    iterations: int = typer.Option(400),
    seed: int = typer.Option(42),
) -> None:
    """Обучить CatBoost, вывести AUC/F1 на отложенных 20%, сохранить models/scam.cbm."""
    from .train import train_model

    if not data.exists():
        console.print(f"[red]Нет датасета {data}[/]. Сначала: uv run scam make-dataset")
        raise typer.Exit(1)
    res = train_model(pd.read_csv(data), out_dir, iterations=iterations, seed=seed)
    t = Table(title=f"Holdout: {res.n_test} объявлений (train {res.n_train})")
    for col in ("ROC-AUC", "F1", "Precision", "Recall", "Accuracy"):
        t.add_column(col, justify="right")
    t.add_row(*(f"{v:.3f}" for v in (res.auc, res.f1, res.precision, res.recall, res.accuracy)))
    console.print(t)
    imp = Table(title="Важность признаков (PredictionValuesChange)")
    imp.add_column("Признак")
    imp.add_column("Важность", justify="right")
    for k, v in list(res.importances.items())[:10]:
        imp.add_row(k, f"{v:.1f}")
    console.print(imp)
    console.print(f"Модель: {res.model_path}")


def _print_report(rep: ScamReport) -> None:
    style = RISK_STYLE[rep.risk_level]
    console.print(
        f"\n[bold]{rep.listing_id}[/]  Scam Score: [{style}]{rep.score:.1f}/100 "
        f"({rep.risk_level.value})[/]"
    )
    for r in rep.reasons:
        console.print(f"  - {r}")
    t = Table(title="Топ-вклады признаков (SHAP, log-odds)", show_edge=False)
    t.add_column("Признак")
    t.add_column("Значение", justify="right")
    t.add_column("Вклад", justify="right")
    for c in rep.contributions[:6]:
        color = "red" if c.contribution > 0 else "green"
        val = f"{c.value:.2f}" if isinstance(c.value, float) else str(c.value)
        t.add_row(c.feature, val, f"[{color}]{c.contribution:+.3f}[/]")
    console.print(t)


@app.command()
def check(
    path: Path = typer.Argument(..., exists=True, dir_okay=False, help="JSON объявления"),
    model: Path = typer.Option(None, help="Путь к модели (по умолчанию models/scam.cbm)"),
    corpus: Path = typer.Option(
        None, help="Папка-корпус для дубликатов (по умолчанию — папка файла)"
    ),
    as_json: bool = typer.Option(False, "--json", help="Вывести ScamReport как JSON"),
) -> None:
    """Проверить одно объявление."""
    from .predict import ScamDetector, build_context

    listing = load_listing(path)
    detector = ScamDetector.load(model or default_model_path())
    ctx = build_context(corpus or path.parent, extra=[listing])
    rep = detector.report(listing, ctx)
    if as_json:
        typer.echo(rep.model_dump_json(indent=2))
    else:
        _print_report(rep)


@app.command()
def batch(
    directory: Path = typer.Argument(..., exists=True, file_okay=False, help="Папка с JSON"),
    model: Path = typer.Option(None, help="Путь к модели"),
    out: Path = typer.Option(None, help="Сохранить отчёты в JSON-файл"),
) -> None:
    """Проверить все объявления в папке (корпус — сама папка)."""
    from .predict import ScamDetector, build_context

    listings = load_corpus(directory)
    if not listings:
        console.print(f"[yellow]В {directory} нет *.json[/]")
        raise typer.Exit(1)
    detector = ScamDetector.load(model or default_model_path())
    ctx = build_context(None, extra=listings)
    reports = sorted((detector.report(x, ctx) for x in listings), key=lambda r: -r.score)
    t = Table(title=f"Scam Score: {directory}")
    t.add_column("Объявление")
    t.add_column("Score", justify="right")
    t.add_column("Риск")
    t.add_column("Главная причина")
    for r in reports:
        style = RISK_STYLE[r.risk_level]
        t.add_row(r.listing_id, f"[{style}]{r.score:.1f}[/]", r.risk_level.value, r.reasons[0])
    console.print(t)
    if out:
        out.write_text(
            json.dumps([r.model_dump(mode="json") for r in reports], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        console.print(f"Отчёты: {out}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Запустить FastAPI (POST /check)."""
    import uvicorn

    uvicorn.run("scam_detector.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
