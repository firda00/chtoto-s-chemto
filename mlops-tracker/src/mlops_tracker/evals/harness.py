"""Evaluation harness: прогоняет task по датасету, считает скореры, логирует всё в трекер.

result = run_eval(answer_question, load_dataset("data/evals/qa.jsonl"),
                  scorers=["contains", "llm_judge"], concurrency=4, retries=1, timeout=30)
print(result.metrics["pass_rate"], result.run_id)
"""

from __future__ import annotations

import asyncio
import inspect
import json
import statistics
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from mlops_tracker.evals.dataset import Case, Dataset, load_dataset
from mlops_tracker.evals.scorers import (
    Score,
    ScoreContext,
    ScorerFn,
    apply_scorer,
    build_scorers,
    scorer_name,
)
from mlops_tracker.llm import LLM
from mlops_tracker.tracker import Tracker, get_tracker

REPORT_TEXT_KEY = "eval_report"


class TaskOutput(BaseModel):
    """Расширенный ответ task: сам ответ + токены/стоимость/трейс (для агентов)."""

    output: Any
    usage: dict[str, int] = Field(default_factory=dict)
    cost_usd: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaseResult(BaseModel):
    id: str
    input: Any
    expected: Any = None
    output: Any = None
    error: str | None = None
    latency_ms: float = 0.0
    attempts: int = 1
    usage: dict[str, int] = Field(default_factory=dict)
    cost_usd: float | None = None
    scores: dict[str, Score] = Field(default_factory=dict)
    passed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    task_metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    run_id: str | None = None
    run_name: str
    task: str
    dataset: str
    dataset_path: str | None = None
    dataset_hash: str
    scorers: list[str]
    params: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    duration_s: float = 0.0
    metrics: dict[str, float] = Field(default_factory=dict)
    cases: list[CaseResult] = Field(default_factory=list)
    report_path: str | None = None

    @property
    def failed_cases(self) -> list[CaseResult]:
        return [c for c in self.cases if not c.passed]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> EvalResult:
        return cls.model_validate_json(text)


class TaskTimeoutError(TimeoutError):
    pass


def task_name(task_fn: Callable[..., Any]) -> str:
    module = getattr(task_fn, "__module__", None) or ""
    name = getattr(task_fn, "__qualname__", None) or type(task_fn).__name__
    return f"{module}:{name}" if module else name


def _call_task(task_fn: Callable[..., Any], case: Case) -> Any:
    """task(input) или task(input, case=...) — если у функции есть параметр `case`
    (нужен доступ к metadata кейса)."""
    try:
        wants_case = "case" in inspect.signature(task_fn).parameters
    except (TypeError, ValueError):
        wants_case = False
    kwargs = {"case": case} if wants_case else {}
    if inspect.iscoroutinefunction(task_fn):
        return asyncio.run(task_fn(case.input, **kwargs))
    result = task_fn(case.input, **kwargs)
    if inspect.isawaitable(result):
        return asyncio.run(_await(result))
    return result


async def _await(aw: Any) -> Any:
    return await aw


def _run_with_timeout(task_fn: Callable[..., Any], case: Case, timeout: float | None) -> Any:
    if not timeout:
        return _call_task(task_fn, case)
    # Отдельный поток на попытку: по таймауту перестаём ждать (поток Python убить нельзя,
    # он доработает в фоне, но результат будет проигнорирован).
    ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlt-task")
    fut = ex.submit(_call_task, task_fn, case)
    try:
        return fut.result(timeout=timeout)
    except FutureTimeout as e:
        raise TaskTimeoutError(f"таймаут {timeout} с") from e
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def _execute_case(
    task_fn: Callable[..., Any],
    case: Case,
    scorers: Sequence[ScorerFn],
    retries: int,
    timeout: float | None,
    retry_backoff: float,
) -> CaseResult:
    attempts, error, raw = 0, None, None
    start = time.perf_counter()
    for attempt in range(retries + 1):
        attempts = attempt + 1
        start = time.perf_counter()
        try:
            raw = _run_with_timeout(task_fn, case, timeout)
            error = None
            break
        except Exception as e:  # noqa: BLE001 — ошибка task фиксируется в результате кейса
            error = f"{type(e).__name__}: {e}"
            if attempt < retries and retry_backoff:
                time.sleep(retry_backoff * (2**attempt))
    latency_ms = (time.perf_counter() - start) * 1000

    output, usage, cost, task_meta = raw, {}, None, {}
    if isinstance(raw, TaskOutput):
        output, usage, cost, task_meta = raw.output, raw.usage, raw.cost_usd, raw.metadata

    result = CaseResult(
        id=case.id,
        input=case.input,
        expected=case.expected,
        output=output,
        error=error,
        latency_ms=round(latency_ms, 3),
        attempts=attempts,
        usage=usage,
        cost_usd=cost,
        metadata=case.metadata,
        task_metadata=task_meta,
    )
    ctx = ScoreContext(
        latency_ms=latency_ms, usage=usage, cost_usd=cost, attempts=attempts, error=error
    )
    for scorer in scorers:
        name = scorer_name(scorer)
        if error is not None:
            result.scores[name] = Score(
                name=name, value=0.0, passed=False, details={"error": error}
            )
        else:
            result.scores[name] = apply_scorer(scorer, case, output, ctx)
    result.passed = error is None and all(s.passed for s in result.scores.values())
    return result


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def aggregate(cases: list[CaseResult]) -> dict[str, float]:
    n = len(cases)
    if n == 0:
        return {"n_cases": 0}
    lat = [c.latency_ms for c in cases]
    metrics: dict[str, float] = {
        "n_cases": n,
        "n_passed": sum(c.passed for c in cases),
        "pass_rate": sum(c.passed for c in cases) / n,
        "error_rate": sum(c.error is not None for c in cases) / n,
        "latency_ms_mean": statistics.fmean(lat),
        "latency_ms_p50": _percentile(lat, 0.5),
        "latency_ms_p95": _percentile(lat, 0.95),
        "retries_total": sum(c.attempts - 1 for c in cases),
        "tokens_prompt": sum(c.usage.get("prompt_tokens", 0) for c in cases),
        "tokens_completion": sum(c.usage.get("completion_tokens", 0) for c in cases),
    }
    costs = [c.cost_usd for c in cases if c.cost_usd is not None]
    if costs:
        metrics["cost_usd_total"] = sum(costs)
    names = sorted({k for c in cases for k in c.scores})
    for name in names:
        scores = [c.scores[name] for c in cases if name in c.scores]
        metrics[f"{name}/mean"] = statistics.fmean(s.value for s in scores)
        metrics[f"{name}/pass_rate"] = sum(s.passed for s in scores) / len(scores)
    return metrics


def run_eval(
    task_fn: Callable[..., Any],
    dataset: Dataset | str | Path,
    scorers: Sequence[ScorerFn | str] | str,
    concurrency: int = 4,
    retries: int = 0,
    timeout: float | None = None,
    retry_backoff: float = 0.0,
    tracker: Tracker | None = None,
    run_name: str | None = None,
    experiment: str = "evals",
    params: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    seed: int | None = None,
    report_dir: str | Path | None = None,
    judge_llm: LLM | None = None,
    log_to_tracker: bool = True,
) -> EvalResult:
    """Прогон оценки.

    * `scorers` — объекты/функции скореров или строки-спеки (`"contains,latency:2000"`);
    * `concurrency` — число параллельных кейсов (потоки);
    * `retries` / `timeout` / `retry_backoff` — устойчивость к флаки-моделям и зависаниям;
    * результат логируется как ран трекера (params, метрики, отчёт JSON в базе),
      а при `report_dir` ещё и сохраняется файлом `eval_<run_id>.json`.
    """
    if not isinstance(dataset, Dataset):
        dataset = load_dataset(dataset)
    resolved: list[ScorerFn] = []
    for s in [scorers] if isinstance(scorers, str) else scorers:
        if isinstance(s, str):
            resolved.extend(build_scorers(s, llm=judge_llm))
        else:
            resolved.append(s)
    names = [scorer_name(s) for s in resolved]
    tname = task_name(task_fn)
    run_name = run_name or f"{tname.split(':')[-1]}@{dataset.name}"

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, concurrency), thread_name_prefix="mlt-eval") as ex:
        futures = [
            ex.submit(_execute_case, task_fn, case, resolved, retries, timeout, retry_backoff)
            for case in dataset.cases
        ]
        cases = [f.result() for f in futures]  # порядок кейсов сохраняется
    duration = time.perf_counter() - t0

    all_params = {
        "task": tname,
        "dataset": dataset.name,
        "dataset_path": dataset.path,
        "dataset_hash": dataset.hash,
        "n_cases": len(dataset),
        "scorers": names,
        "concurrency": concurrency,
        "retries": retries,
        "timeout": timeout,
        **(params or {}),
    }
    result = EvalResult(
        run_name=run_name,
        task=tname,
        dataset=dataset.name,
        dataset_path=dataset.path,
        dataset_hash=dataset.hash,
        scorers=names,
        params=all_params,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        duration_s=round(duration, 3),
        metrics=aggregate(cases),
        cases=cases,
    )

    if log_to_tracker:
        tracker = tracker or get_tracker()
        with tracker.start_run(
            run_name,
            params=all_params,
            tags=["eval", *(tags or [])],
            experiment=experiment,
            seed=seed,
        ) as run:
            result.run_id = run.id
            run.log_metrics({**result.metrics, "duration_s": duration})
            for i, c in enumerate(cases):  # покейсовые кривые: step = номер кейса
                run.log_metric("case/passed", float(c.passed), step=i)
                run.log_metric("case/latency_ms", c.latency_ms, step=i)
            if report_dir:
                result.report_path = str(Path(report_dir) / f"eval_{run.id}.json")
            run.log_text(REPORT_TEXT_KEY, result.to_json())
            if report_dir:
                path = Path(result.report_path)  # type: ignore[arg-type]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(result.to_json(), encoding="utf-8")
                run.log_artifact(path)
    elif report_dir:
        path = Path(report_dir) / f"eval_{run_name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        result.report_path = str(path)
        path.write_text(result.to_json(), encoding="utf-8")
    return result


def load_eval_result(store_or_tracker: Any, run_id: str) -> EvalResult:
    """Достаёт JSON-отчёт eval-рана из базы трекера."""
    store = getattr(store_or_tracker, "store", store_or_tracker)
    text = store.get_text(run_id, REPORT_TEXT_KEY)
    if text is None:
        raise ValueError(f"Ран {run_id} не является eval-раном (нет отчёта '{REPORT_TEXT_KEY}')")
    return EvalResult.from_json(text)


def load_eval_json(path: str | Path) -> EvalResult:
    return EvalResult.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
