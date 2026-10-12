"""Отчёты по eval-ранам: Markdown и самодостаточный HTML (jinja2) + diff двух ранов."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from jinja2 import Environment, select_autoescape
from pydantic import BaseModel, Field

from mlops_tracker.evals.harness import EvalResult

CaseStatus = Literal["regression", "fixed", "still_failing", "still_passing", "added", "removed"]

# Метрики, где «больше» — хуже (для подсветки дельт).
LOWER_IS_BETTER = ("latency", "error_rate", "cost", "tokens", "retries", "duration")


class CaseDiff(BaseModel):
    id: str
    status: CaseStatus
    base_passed: bool | None = None
    new_passed: bool | None = None
    score_deltas: dict[str, float] = Field(default_factory=dict)
    base_output: Any = None
    new_output: Any = None


class MetricDelta(BaseModel):
    base: float | None
    new: float | None
    delta: float | None
    better: bool | None = None


class EvalDiff(BaseModel):
    base_run: str | None
    new_run: str | None
    base_name: str
    new_name: str
    dataset_changed: bool
    metrics: dict[str, MetricDelta]
    cases: list[CaseDiff]

    @property
    def regressions(self) -> list[CaseDiff]:
        return [c for c in self.cases if c.status == "regression"]

    @property
    def fixed(self) -> list[CaseDiff]:
        return [c for c in self.cases if c.status == "fixed"]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.cases:
            out[c.status] = out.get(c.status, 0) + 1
        return out


def _lower_is_better(metric: str) -> bool:
    return any(token in metric for token in LOWER_IS_BETTER)


def diff_results(base: EvalResult, new: EvalResult) -> EvalDiff:
    """Покейсовое сравнение: регрессии (passed→failed), исправления, добавленные/удалённые кейсы,
    дельты скореров и агрегированных метрик."""
    metrics: dict[str, MetricDelta] = {}
    for key in sorted(set(base.metrics) | set(new.metrics)):
        b, n = base.metrics.get(key), new.metrics.get(key)
        delta = n - b if b is not None and n is not None else None
        better = None
        if delta:
            better = delta < 0 if _lower_is_better(key) else delta > 0
        metrics[key] = MetricDelta(base=b, new=n, delta=delta, better=better)

    base_by_id = {c.id: c for c in base.cases}
    new_by_id = {c.id: c for c in new.cases}
    ordered_ids = [c.id for c in new.cases] + [c.id for c in base.cases if c.id not in new_by_id]
    cases: list[CaseDiff] = []
    for cid in ordered_ids:
        b, n = base_by_id.get(cid), new_by_id.get(cid)
        if b is None:
            cases.append(CaseDiff(id=cid, status="added", new_passed=n.passed, new_output=n.output))
            continue
        if n is None:
            cases.append(
                CaseDiff(id=cid, status="removed", base_passed=b.passed, base_output=b.output)
            )
            continue
        if b.passed and not n.passed:
            status: CaseStatus = "regression"
        elif not b.passed and n.passed:
            status = "fixed"
        else:
            status = "still_passing" if n.passed else "still_failing"
        deltas = {
            name: round(n.scores[name].value - b.scores[name].value, 4)
            for name in n.scores
            if name in b.scores
        }
        cases.append(
            CaseDiff(
                id=cid,
                status=status,
                base_passed=b.passed,
                new_passed=n.passed,
                score_deltas=deltas,
                base_output=b.output,
                new_output=n.output,
            )
        )
    return EvalDiff(
        base_run=base.run_id,
        new_run=new.run_id,
        base_name=base.run_name,
        new_name=new.run_name,
        dataset_changed=base.dataset_hash != new.dataset_hash,
        metrics=metrics,
        cases=cases,
    )


# ------------------------------------------------------------------ helpers
def fmt_value(v: Any, limit: int = 120) -> str:
    if v is None:
        return "—"
    text = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def fmt_metric(key: str, v: float | None) -> str:
    if v is None:
        return "—"
    if "rate" in key or key.endswith("/mean"):
        return f"{v:.1%}" if "rate" in key else f"{v:.3f}"
    if "cost" in key:
        return f"${v:.6f}"
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.2f}"


def fmt_delta(key: str, d: float | None) -> str:
    if d is None:
        return "—"
    if "rate" in key:
        return f"{d:+.1%}"
    if "cost" in key:
        return f"{d:+.6f}"
    return f"{d:+.3f}" if not float(d).is_integer() else f"{int(d):+d}"


def _md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


KEY_METRICS = (
    "pass_rate",
    "n_passed",
    "n_cases",
    "error_rate",
    "latency_ms_p50",
    "latency_ms_p95",
    "cost_usd_total",
    "tokens_prompt",
    "tokens_completion",
)

STATUS_LABELS = {
    "regression": "РЕГРЕССИЯ",
    "fixed": "исправлено",
    "still_failing": "провал (без изм.)",
    "still_passing": "ок (без изм.)",
    "added": "новый кейс",
    "removed": "удалён",
}


# ----------------------------------------------------------------- markdown
def render_markdown(result: EvalResult, diff: EvalDiff | None = None) -> str:
    m = result.metrics
    lines = [
        f"# Eval-отчёт: {result.run_name}",
        "",
        f"- **Ран:** `{result.run_id or '—'}`  ",
        f"- **Task:** `{result.task}`  ",
        f"- **Датасет:** `{result.dataset}` (hash `{result.dataset_hash}`, {len(result.cases)} "
        "кейсов)  ",
        f"- **Скореры:** {', '.join(result.scorers)}  ",
        f"- **Создан:** {result.created_at}, длительность {result.duration_s:.2f} с",
        "",
        "## Сводка",
        "",
        "| Метрика | Значение |",
        "|---|---|",
    ]
    for key in KEY_METRICS:
        if key in m:
            lines.append(f"| {key} | {fmt_metric(key, m[key])} |")
    lines += ["", "## Скореры", "", "| Скорер | Средняя оценка | Pass rate |", "|---|---|---|"]
    for name in result.scorers:
        lines.append(
            f"| {name} | {fmt_metric(name + '/mean', m.get(name + '/mean'))} "
            f"| {fmt_metric('pass_rate', m.get(name + '/pass_rate'))} |"
        )
    header = "| Кейс | Статус | " + " | ".join(result.scorers) + " | Латентность, мс | Ответ |"
    lines += ["", "## Кейсы", "", header, "|" + "---|" * (len(result.scorers) + 4)]
    for c in result.cases:
        status = "PASS" if c.passed else ("ERROR" if c.error else "FAIL")
        cells = [f"{c.scores[s].value:.2f}" if s in c.scores else "—" for s in result.scorers]
        out = c.error if c.error else fmt_value(c.output, 80)
        lines.append(
            f"| {_md(c.id)} | {status} | {' | '.join(cells)} | {c.latency_ms:.1f} | {_md(out)} |"
        )
    failed = result.failed_cases
    if failed:
        lines += ["", "## Провалы — детали", ""]
        for c in failed:
            lines.append(f"### {c.id}")
            lines.append(f"- **Вход:** {_md(fmt_value(c.input, 300))}")
            lines.append(f"- **Ожидалось:** {_md(fmt_value(c.expected, 300))}")
            lines.append(f"- **Получено:** {_md(fmt_value(c.output, 300))}")
            if c.error:
                lines.append(f"- **Ошибка:** `{c.error}`")
            for s in c.scores.values():
                if not s.passed:
                    lines.append(f"- `{s.name}` = {s.value:.2f}: {_md(fmt_value(s.details, 300))}")
            lines.append("")
    if diff is not None:
        lines += render_diff_markdown(diff)
    return "\n".join(lines).rstrip() + "\n"


def render_diff_markdown(diff: EvalDiff) -> list[str]:
    counts = diff.counts()
    lines = [
        "",
        f"## Сравнение: `{diff.base_name}` → `{diff.new_name}`",
        "",
        f"Базовый ран `{diff.base_run}`, новый ран `{diff.new_run}`. "
        f"Регрессий: **{counts.get('regression', 0)}**, исправлено: "
        f"**{counts.get('fixed', 0)}**.",
    ]
    if diff.dataset_changed:
        lines.append("")
        lines.append(
            "> **Внимание:** датасеты отличаются (разный hash) — сравнение может быть некорректным."
        )
    lines += ["", "| Метрика | База | Новый | Δ |", "|---|---|---|---|"]
    for key in KEY_METRICS:
        if key in diff.metrics:
            d = diff.metrics[key]
            mark = "" if d.better is None else (" (лучше)" if d.better else " (хуже)")
            lines.append(
                f"| {key} | {fmt_metric(key, d.base)} | {fmt_metric(key, d.new)} "
                f"| {fmt_delta(key, d.delta)}{mark} |"
            )
    changed = [c for c in diff.cases if c.status not in ("still_passing", "still_failing")]
    if changed:
        lines += ["", "| Кейс | Изменение | Было | Стало |", "|---|---|---|---|"]
        for c in changed:
            lines.append(
                f"| {_md(c.id)} | {STATUS_LABELS[c.status]} | {_md(fmt_value(c.base_output, 60))} "
                f"| {_md(fmt_value(c.new_output, 60))} |"
            )
    return lines


# --------------------------------------------------------------------- html
def _env() -> Environment:
    env = Environment(
        autoescape=select_autoescape(default=True), trim_blocks=True, lstrip_blocks=True
    )
    env.filters["fv"] = fmt_value
    env.filters["fm"] = lambda v, key: fmt_metric(key, v)
    env.filters["fd"] = lambda v, key: fmt_delta(key, v)
    env.filters["pretty"] = lambda v: (
        v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, indent=2, default=str)
    )
    return env


def _template_source() -> str:
    return (resources.files("mlops_tracker") / "templates" / "report.html.j2").read_text(
        encoding="utf-8"
    )


def render_html(result: EvalResult, diff: EvalDiff | None = None) -> str:
    template = _env().from_string(_template_source())
    return template.render(
        r=result,
        diff=diff,
        key_metrics=[k for k in KEY_METRICS if k in result.metrics],
        status_labels=STATUS_LABELS,
    )


def write_report(
    result: EvalResult,
    html_path: str | Path | None = None,
    md_path: str | Path | None = None,
    baseline: EvalResult | None = None,
) -> dict[str, Path]:
    diff = diff_results(baseline, result) if baseline is not None else None
    written: dict[str, Path] = {}
    for kind, path, render in (
        ("html", html_path, render_html),
        ("md", md_path, render_markdown),
    ):
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(render(result, diff), encoding="utf-8")
            written[kind] = p
    return written
