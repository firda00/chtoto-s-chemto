from __future__ import annotations

from pathlib import Path

import pytest

from mlops_tracker.evals.dataset import Dataset
from mlops_tracker.evals.harness import EvalResult, run_eval
from mlops_tracker.evals.report import diff_results, render_html, render_markdown, write_report
from mlops_tracker.evals.scorers import Contains, LLMJudge
from mlops_tracker.llm import MockLLM
from mlops_tracker.tracker import Tracker

CASES = [
    {"id": "geo", "input": "Столица Франции?", "expected": "Париж"},
    {"id": "math", "input": "2 + 2 * 2", "expected": "6"},
    {"id": "lit", "input": "Автор «Войны и мира»?", "expected": "Толстой"},
    {"id": "html", "input": "<script>", "expected": "<b>жирный</b>"},
]

V1 = {"geo": "Париж", "math": "8", "lit": "Толстой", "html": "<b>жирный</b>"}
V2 = {"geo": "Париж", "math": "6", "lit": "Не знаю", "html": "<b>жирный</b>"}


@pytest.fixture
def two_runs(tracker: Tracker) -> tuple[EvalResult, EvalResult]:
    ds = Dataset.from_records(CASES, name="mini")
    scorers = [Contains(), LLMJudge(llm=MockLLM(), threshold=3)]
    base = run_eval(lambda q, case: V1[case.id], ds, scorers, tracker=tracker, run_name="v1")
    new = run_eval(lambda q, case: V2[case.id], ds, scorers, tracker=tracker, run_name="v2")
    return base, new


def test_diff_detects_regressions_and_fixes(two_runs) -> None:
    base, new = two_runs
    diff = diff_results(base, new)
    assert [c.id for c in diff.regressions] == ["lit"]
    assert [c.id for c in diff.fixed] == ["math"]
    assert diff.counts() == {"still_passing": 2, "fixed": 1, "regression": 1}
    lit = next(c for c in diff.cases if c.id == "lit")
    assert lit.score_deltas["contains"] == -1.0
    assert lit.base_output == "Толстой" and lit.new_output == "Не знаю"
    assert diff.metrics["pass_rate"].delta == 0.0
    assert not diff.dataset_changed


def test_diff_added_removed_and_dataset_change(tracker: Tracker) -> None:
    a = run_eval(
        lambda q: q,
        Dataset.from_records([{"id": "x", "input": "1", "expected": "1"}]),
        "exact_match",
        tracker=tracker,
    )
    b = run_eval(
        lambda q: q,
        Dataset.from_records([{"id": "y", "input": "2", "expected": "3"}]),
        "exact_match",
        tracker=tracker,
    )
    diff = diff_results(a, b)
    assert {c.id: c.status for c in diff.cases} == {"y": "added", "x": "removed"}
    assert diff.dataset_changed
    assert diff.metrics["pass_rate"].better is False


def test_markdown_report(two_runs) -> None:
    base, new = two_runs
    md = render_markdown(new, diff_results(base, new))
    assert md.startswith("# Eval-отчёт: v2")
    assert "| pass_rate | 75.0% |" in md
    assert "| contains |" in md and "| llm_judge |" in md
    assert "## Провалы — детали" in md and "### lit" in md
    assert "РЕГРЕССИЯ" in md and "исправлено" in md
    assert f"`{base.run_id}`" in md


def test_html_report_is_standalone_and_escaped(two_runs, tmp_path: Path) -> None:
    base, new = two_runs
    html = render_html(new, diff_results(base, new))
    assert html.startswith("<!doctype html>")
    assert "Eval-отчёт: v2" in html
    assert 'class="regression"' in html and 'class="fixed"' in html
    assert "РЕГРЕССИЯ" in html
    assert "&lt;script&gt;" in html and "<script>" not in html  # автоэкранирование
    assert "http://" not in html and "https://" not in html  # без внешних ресурсов
    for c in new.cases:
        assert c.id in html

    paths = write_report(
        new, html_path=tmp_path / "r.html", md_path=tmp_path / "r.md", baseline=base
    )
    assert paths["html"].read_text("utf-8").count("<tr") > len(CASES)
    assert "Сравнение" in paths["md"].read_text("utf-8")


def test_html_without_diff(two_runs) -> None:
    _, new = two_runs
    html = render_html(new)
    assert "Сравнение с базовым раном" not in html
    assert "llm_judge" in html
