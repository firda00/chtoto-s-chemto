"""CLI `mlt`: просмотр ранов, сравнение, запуск оценки и генерация отчётов."""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from mlops_tracker.evals.dataset import load_dataset
from mlops_tracker.evals.harness import load_eval_result, run_eval
from mlops_tracker.evals.report import diff_results, write_report
from mlops_tracker.evals.scorers import build_scorers
from mlops_tracker.llm import get_llm
from mlops_tracker.store import RunNotFoundError, Store
from mlops_tracker.tracker import Tracker

app = typer.Typer(
    help="mlops-tracker: трекер экспериментов + evaluation harness", no_args_is_help=True
)
runs_app = typer.Typer(help="Просмотр и сравнение ранов", no_args_is_help=True)
eval_app = typer.Typer(help="Запуск оценки", no_args_is_help=True)
app.add_typer(runs_app, name="runs")
app.add_typer(eval_app, name="eval")

console = Console()


class State:
    db: Path | None = None


state = State()


@app.callback()
def _main(
    db: Annotated[
        Path | None,
        typer.Option("--db", help="Путь к SQLite (по умолчанию $MLT_DB или mlruns/mlt.db)"),
    ] = None,
) -> None:
    state.db = db


def _store() -> Store:
    return Store(state.db)


def _fail(msg: str) -> None:
    console.print(f"[red]Ошибка:[/red] {msg}")
    raise typer.Exit(1)


def _ts(t: float | None) -> str:
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S") if t else "—"


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}"
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


# -------------------------------------------------------------------- runs
@runs_app.command("list")
def runs_list(
    experiment: Annotated[str | None, typer.Option("--experiment", "-e")] = None,
    tag: Annotated[str | None, typer.Option("--tag", "-t")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
) -> None:
    """Список ранов (новые сверху)."""
    runs = _store().list_runs(experiment=experiment, tag=tag, limit=limit)
    if not runs:
        console.print("Ранов пока нет.")
        return
    table = Table(title=f"Раны ({len(runs)})")
    table.add_column("id", no_wrap=True, min_width=12)
    table.add_column("name", min_width=14)
    table.add_column("experiment")
    table.add_column("status")
    table.add_column("started", no_wrap=True)
    table.add_column("pass_rate", no_wrap=True)
    table.add_column("tags")
    for r in runs:
        pr = r.last_metrics().get("pass_rate")
        style = {"failed": "red", "running": "yellow"}.get(r.status, "green")
        table.add_row(
            r.id,
            r.name,
            r.experiment,
            f"[{style}]{r.status}[/{style}]",
            _ts(r.start_time),
            f"{pr:.1%}" if pr is not None else "—",
            ",".join(r.tags),
        )
    console.print(table)


@runs_app.command("show")
def runs_show(run_id: str) -> None:
    """Детали рана: параметры, метрики, окружение, артефакты."""
    try:
        r = _store().get_run(run_id)
    except RunNotFoundError as e:
        _fail(str(e.args[0]))
        return
    console.print(f"[bold]{r.name}[/bold]  id={r.id}  experiment={r.experiment}  status={r.status}")
    dur = f"{r.duration_s:.2f} с" if r.duration_s is not None else "—"
    console.print(
        f"start={_ts(r.start_time)}  duration={dur}  seed={r.seed}  "
        f"git={r.git_commit or '—'}  tags={','.join(r.tags) or '—'}"
    )
    env = r.env
    if env:
        console.print(
            f"python={env.get('python')}  platform={env.get('platform')}  "
            f"packages={len(env.get('packages', {}))}"
        )
    if r.params:
        t = Table(title="Параметры")
        t.add_column("key")
        t.add_column("value")
        for k, v in r.params.items():
            t.add_row(k, _fmt(v))
        console.print(t)
    if r.metrics:
        t = Table(title="Метрики")
        for col in ("key", "last", "steps", "min", "max"):
            t.add_column(col)
        for k, pts in r.metrics.items():
            vals = [p.value for p in pts]
            t.add_row(k, _fmt(vals[-1]), str(len(vals)), _fmt(min(vals)), _fmt(max(vals)))
        console.print(t)
    if r.artifacts:
        console.print("Артефакты:")
        for name, path in r.artifacts.items():
            console.print(f"  {name} -> {path}")
    if r.texts:
        console.print(f"Тексты: {', '.join(r.texts)}")
    if r.error:
        console.print(f"[red]Ошибка рана:[/red]\n{r.error}")


@runs_app.command("compare")
def runs_compare(run_a: str, run_b: str) -> None:
    """Сравнение двух ранов: параметры, метрики, а для eval-ранов — регрессии по кейсам."""
    store = _store()
    try:
        cmp = store.compare_runs(run_a, run_b)
    except RunNotFoundError as e:
        _fail(str(e.args[0]))
        return
    a, b = cmp["a"], cmp["b"]
    console.print(f"A: {a['name']} ({a['id']})   B: {b['name']} ({b['id']})")
    t = Table(title="Параметры (отличающиеся)")
    for col in ("key", "A", "B"):
        t.add_column(col)
    for k, v in cmp["params"].items():
        if v["changed"]:
            t.add_row(k, _fmt(v["a"]), _fmt(v["b"]))
    console.print(t)
    t = Table(title="Метрики (последние значения)")
    for col in ("key", "A", "B", "Δ"):
        t.add_column(col)
    for k, v in cmp["metrics"].items():
        if k.startswith("case/"):
            continue
        d = v["delta"]
        t.add_row(k, _fmt(v["a"]), _fmt(v["b"]), f"{d:+.4g}" if d is not None else "—")
    console.print(t)
    try:
        diff = diff_results(load_eval_result(store, a["id"]), load_eval_result(store, b["id"]))
    except ValueError:
        return
    counts = diff.counts()
    console.print(
        f"Кейсы: регрессий [red]{counts.get('regression', 0)}[/red], "
        f"исправлено [green]{counts.get('fixed', 0)}[/green]"
    )
    for c in diff.cases:
        if c.status in ("regression", "fixed"):
            color = "red" if c.status == "regression" else "green"
            console.print(f"  [{color}]{c.status}[/{color}] {c.id}")


@runs_app.command("best")
def runs_best(
    metric: Annotated[str, typer.Option("--metric", "-m")] = "pass_rate",
    mode: Annotated[str, typer.Option("--mode")] = "max",
    experiment: Annotated[str | None, typer.Option("--experiment", "-e")] = None,
) -> None:
    """Лучший ран по метрике."""
    found = _store().best_run(metric, mode=mode, experiment=experiment)  # type: ignore[arg-type]
    if not found:
        _fail(f"Нет завершённых ранов с метрикой {metric}")
        return
    run, value = found
    console.print(f"Лучший по {metric} ({mode}): {run.name} ({run.id}) = {value:.4g}")


# -------------------------------------------------------------------- eval
def import_task(spec: str) -> Any:
    """`package.module:function` → callable. Текущая папка добавляется в sys.path."""
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise typer.BadParameter("ожидается формат module.path:function", param_hint="--task")
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as e:
        raise typer.BadParameter(f"в {module_name} нет {attr}", param_hint="--task") from e


@eval_app.command("run")
def eval_run(
    dataset: Annotated[Path, typer.Option("--dataset", "-d", help="JSONL/JSON/YAML с кейсами")],
    task: Annotated[str, typer.Option("--task", "-t", help="module.path:function")],
    scorers: Annotated[
        str, typer.Option("--scorers", "-s", help="через запятую, напр. contains,llm_judge:4")
    ] = "contains",
    concurrency: Annotated[int, typer.Option("--concurrency", "-c")] = 4,
    retries: Annotated[int, typer.Option("--retries")] = 0,
    timeout: Annotated[float | None, typer.Option("--timeout", help="секунд на кейс")] = None,
    name: Annotated[str | None, typer.Option("--name", "-n")] = None,
    experiment: Annotated[str, typer.Option("--experiment", "-e")] = "evals",
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    llm: Annotated[str, typer.Option("--llm", help="судья: auto | mock | openai")] = "auto",
    out: Annotated[Path | None, typer.Option("--out", "-o", help="папка для JSON-отчёта")] = None,
    html: Annotated[Path | None, typer.Option("--html", help="сразу записать HTML-отчёт")] = None,
    fail_under: Annotated[
        float | None, typer.Option("--fail-under", help="exit 1, если pass_rate ниже порога (CI)")
    ] = None,
) -> None:
    """Прогнать task по датасету со скорерами и залогировать ран."""
    try:
        ds = load_dataset(dataset)
    except (OSError, ValueError) as e:
        _fail(str(e))
        return
    task_fn = import_task(task)
    judge = get_llm(llm) if "llm_judge" in scorers else None
    try:
        scorer_objs = build_scorers(scorers, llm=judge)
    except ValueError as e:
        _fail(str(e))
        return
    console.print(
        f"Датасет [bold]{ds.name}[/bold] ({len(ds)} кейсов, hash {ds.hash}), task {task}, "
        f"скореры: {', '.join(s.name for s in scorer_objs)}"
        + (f", судья: {judge.model}" if judge else "")
    )
    result = run_eval(
        task_fn,
        ds,
        scorer_objs,
        concurrency=concurrency,
        retries=retries,
        timeout=timeout,
        tracker=Tracker(state.db),
        run_name=name,
        experiment=experiment,
        seed=seed,
        params={"judge_model": judge.model} if judge else None,
        report_dir=out,
    )
    t = Table(title=f"Результат: {result.run_name}")
    for col in ("кейс", "статус", *result.scorers, "мс"):
        t.add_column(col)
    for c in result.cases:
        st = (
            "[green]PASS[/green]"
            if c.passed
            else ("[red]ERROR[/red]" if c.error else "[red]FAIL[/red]")
        )
        cells = [f"{c.scores[s].value:.2f}" if s in c.scores else "—" for s in result.scorers]
        t.add_row(c.id, st, *cells, f"{c.latency_ms:.1f}")
    console.print(t)
    m = result.metrics
    console.print(
        f"pass_rate={m['pass_rate']:.1%} ({int(m['n_passed'])}/{int(m['n_cases'])}), "
        f"p50={m['latency_ms_p50']:.1f} мс, p95={m['latency_ms_p95']:.1f} мс"
    )
    console.print(f"run_id: [bold]{result.run_id}[/bold]")
    if result.report_path:
        console.print(f"JSON-отчёт: {result.report_path}")
    if html:
        write_report(result, html_path=html)
        console.print(f"HTML-отчёт: {html}")
    if fail_under is not None and m["pass_rate"] < fail_under:
        console.print(f"[red]pass_rate {m['pass_rate']:.1%} ниже порога {fail_under:.1%}[/red]")
        raise typer.Exit(1)


# ------------------------------------------------------------------ report
@app.command("report")
def report(
    run_id: str,
    html: Annotated[Path | None, typer.Option("--html", help="путь к HTML-отчёту")] = None,
    md: Annotated[Path | None, typer.Option("--md", help="путь к Markdown-отчёту")] = None,
    baseline: Annotated[
        str | None, typer.Option("--baseline", "-b", help="ран для сравнения (регрессии)")
    ] = None,
) -> None:
    """Отчёт по eval-рану (Markdown в консоль, если не указаны --html/--md)."""
    store = _store()
    try:
        result = load_eval_result(store, run_id)
        base = load_eval_result(store, baseline) if baseline else None
    except (RunNotFoundError, ValueError) as e:
        _fail(str(e.args[0]) if e.args else str(e))
        return
    if not html and not md:
        from mlops_tracker.evals.report import render_markdown

        diff = diff_results(base, result) if base else None
        typer.echo(render_markdown(result, diff))
        return
    for kind, path in write_report(result, html_path=html, md_path=md, baseline=base).items():
        console.print(f"{kind.upper()}-отчёт: {path}")


def main() -> None:
    # Кириллица в консоли Windows (cp866/cp1251) — переводим вывод в UTF-8.
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    app()


if __name__ == "__main__":
    main()
