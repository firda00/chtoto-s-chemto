from __future__ import annotations

import re
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from mlops_tracker import cli
from mlops_tracker.store import Store

ROOT = Path(__file__).resolve().parents[1]
runner = CliRunner()


@pytest.fixture(autouse=True)
def _wide_console(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "console", Console(width=250, no_color=True))
    monkeypatch.chdir(ROOT)  # чтобы импортировался examples.tasks


def invoke(db: Path, *args: str):
    result = runner.invoke(cli.app, ["--db", str(db), *args])
    assert result.exit_code == 0, result.output + repr(result.exception)
    return result


def run_id_from(output: str) -> str:
    return re.search(r"run_id: (\w+)", output).group(1)


def test_cli_end_to_end(db_path: Path, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    r1 = invoke(
        db_path,
        "eval",
        "run",
        "--dataset",
        "data/evals/qa.jsonl",
        "--task",
        "examples.tasks:answer_question",
        "--scorers",
        "contains,llm_judge",
        "--name",
        "baseline",
        "--out",
        str(out_dir),
        "--llm",
        "mock",
    )
    assert "pass_rate=" in r1.output and "mock-llm" in r1.output
    base_id = run_id_from(r1.output)
    assert (out_dir / f"eval_{base_id}.json").exists()

    r2 = invoke(
        db_path,
        "eval",
        "run",
        "-d",
        "data/evals/qa.jsonl",
        "-t",
        "examples.tasks:mock_agent",
        "-s",
        "contains,cost:0.01",
        "-n",
        "agent",
        "--html",
        str(out_dir / "agent.html"),
    )
    agent_id = run_id_from(r2.output)
    assert (out_dir / "agent.html").exists()

    listing = invoke(db_path, "runs", "list").output
    assert base_id in listing and agent_id in listing and "baseline" in listing

    show = invoke(db_path, "runs", "show", base_id[:6]).output
    assert "Параметры" in show and "pass_rate" in show and "dataset_hash" in show

    cmp = invoke(db_path, "runs", "compare", base_id, agent_id).output
    assert "регрессий 1" in cmp and "исправлено 1" in cmp
    assert "qa-mlops" in cmp and "qa-math-precedence" in cmp

    best = invoke(db_path, "runs", "best", "--metric", "pass_rate").output
    assert "Лучший по pass_rate" in best

    html = out_dir / "report.html"
    md = out_dir / "report.md"
    invoke(db_path, "report", agent_id, "--html", str(html), "--md", str(md), "-b", base_id)
    assert "РЕГРЕССИЯ" in html.read_text("utf-8")
    assert "qa-mlops" in md.read_text("utf-8")

    printed = invoke(db_path, "report", base_id).output
    assert printed.startswith("# Eval-отчёт: baseline")


def test_cli_errors(db_path: Path) -> None:
    Store(db_path)
    r = runner.invoke(cli.app, ["--db", str(db_path), "runs", "show", "nope"])
    assert r.exit_code == 1 and "не найден" in r.output
    r = runner.invoke(
        cli.app,
        [
            "--db",
            str(db_path),
            "eval",
            "run",
            "-d",
            "missing.jsonl",
            "-t",
            "examples.tasks:answer_question",
        ],
    )
    assert r.exit_code == 1
    r = runner.invoke(
        cli.app,
        [
            "--db",
            str(db_path),
            "eval",
            "run",
            "-d",
            "data/evals/qa.jsonl",
            "-t",
            "examples.tasks:answer_question",
            "-s",
            "bogus",
        ],
    )
    assert r.exit_code == 1 and "Неизвестный скорер" in r.output
    empty = runner.invoke(cli.app, ["--db", str(db_path), "runs", "list"])
    assert "Ранов пока нет" in empty.output


def test_cli_fail_under_for_ci(db_path: Path) -> None:
    r = runner.invoke(
        cli.app,
        [
            "--db",
            str(db_path),
            "eval",
            "run",
            "-d",
            "data/evals/qa.jsonl",
            "-t",
            "examples.tasks:answer_question",
            "--fail-under",
            "0.95",
        ],
    )
    assert r.exit_code == 1 and "ниже порога" in r.output
