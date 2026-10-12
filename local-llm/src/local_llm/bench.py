"""Бенчмарк любого OpenAI-совместимого endpoint (Ollama /v1, vLLM, OpenRouter...).

Метрики на запрос (через стриминг SSE):
- TTFT          — время от отправки запроса до первого непустого токена;
- decode tok/s  — (N − 1) / (t_последнего − t_первого токена), скорость генерации;
- e2e tok/s     — N / полное время ответа;
- latency       — полное время ответа.

N берётся из `usage.completion_tokens` (stream_options.include_usage), а если сервер его
не прислал — по числу контентных чанков (Ollama и vLLM шлют ~1 токен на чанк).
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field
from rich.table import Table

Clock = Callable[[], float]


class BenchConfig(BaseModel):
    base_url: str
    model: str
    api_key: str = Field(default="local", exclude=True, repr=False)
    label: str | None = None
    max_tokens: int = Field(default=256, gt=0)
    temperature: float = Field(default=0.0, ge=0)
    repeats: int = Field(default=1, ge=1)
    warmup: int = Field(default=1, ge=0)
    timeout: float = Field(default=300.0, gt=0)
    extra_body: dict[str, Any] = Field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.label or self.model


class RequestResult(BaseModel):
    prompt_idx: int
    repeat: int
    ok: bool
    ttft_s: float | None = None
    total_s: float
    completion_tokens: int = 0
    prompt_tokens: int | None = None
    decode_tps: float | None = None
    e2e_tps: float | None = None
    tokens_from_usage: bool = False
    error: str | None = None


class Stats(BaseModel):
    mean: float | None = None
    p50: float | None = None
    p95: float | None = None
    min: float | None = None
    max: float | None = None

    @classmethod
    def of(cls, values: Sequence[float | None], digits: int = 4) -> Stats:
        vals = sorted(v for v in values if v is not None)
        if not vals:
            return cls()
        r = lambda x: round(x, digits)  # noqa: E731
        return cls(
            mean=r(sum(vals) / len(vals)),
            p50=r(percentile(vals, 50)),
            p95=r(percentile(vals, 95)),
            min=r(vals[0]),
            max=r(vals[-1]),
        )


class BenchSummary(BaseModel):
    label: str
    base_url: str
    model: str
    n_requests: int
    n_errors: int
    total_completion_tokens: int
    ttft_ms: Stats
    decode_tps: Stats
    e2e_tps: Stats
    latency_s: Stats


class BenchRun(BaseModel):
    config: BenchConfig
    started_at: str
    summary: BenchSummary
    results: list[RequestResult]


def percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Перцентиль с линейной интерполяцией (как numpy по умолчанию)."""
    if not sorted_vals:
        raise ValueError("пустая выборка")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo))


def load_prompts(path: str | Path) -> list[str]:
    """Один промпт на строку; пустые строки и строки с `#` пропускаются."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    prompts = [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    if not prompts:
        raise ValueError(f"{path}: нет промптов")
    return prompts


def _endpoint(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def measure_request(
    client: httpx.Client,
    cfg: BenchConfig,
    prompt: str,
    *,
    clock: Clock = time.perf_counter,
    prompt_idx: int = 0,
    repeat: int = 0,
) -> RequestResult:
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        **cfg.extra_body,
    }
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Accept": "text/event-stream"}
    t0 = clock()
    first: float | None = None
    last: float | None = None
    chunks = 0
    usage: dict[str, Any] | None = None
    try:
        with client.stream("POST", _endpoint(cfg.base_url), json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                resp.read()
                raise httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}: {resp.text[:200]}",
                    request=resp.request,
                    response=resp,
                )
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or delta.get("reasoning_content") or ""
                    if text:
                        now = clock()
                        if first is None:
                            first = now
                        last = now
                        chunks += 1
        end = clock()
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        return RequestResult(
            prompt_idx=prompt_idx,
            repeat=repeat,
            ok=False,
            total_s=round(clock() - t0, 6),
            error=f"{type(exc).__name__}: {exc}"[:300],
        )

    total = end - t0
    from_usage = bool(usage and usage.get("completion_tokens"))
    n_tokens = int(usage["completion_tokens"]) if from_usage and usage else chunks
    decode = None
    if first is not None and last is not None and last > first and n_tokens > 1:
        decode = (n_tokens - 1) / (last - first)
    return RequestResult(
        prompt_idx=prompt_idx,
        repeat=repeat,
        ok=first is not None,
        ttft_s=round(first - t0, 6) if first is not None else None,
        total_s=round(total, 6),
        completion_tokens=n_tokens,
        prompt_tokens=(usage or {}).get("prompt_tokens"),
        decode_tps=round(decode, 4) if decode is not None else None,
        e2e_tps=round(n_tokens / total, 4) if total > 0 and n_tokens else None,
        tokens_from_usage=from_usage,
        error=None if first is not None else "пустой ответ",
    )


def summarize(cfg: BenchConfig, results: list[RequestResult]) -> BenchSummary:
    ok = [r for r in results if r.ok]
    return BenchSummary(
        label=cfg.name,
        base_url=cfg.base_url,
        model=cfg.model,
        n_requests=len(results),
        n_errors=len(results) - len(ok),
        total_completion_tokens=sum(r.completion_tokens for r in ok),
        ttft_ms=Stats.of([r.ttft_s * 1000 for r in ok if r.ttft_s is not None], digits=2),
        decode_tps=Stats.of([r.decode_tps for r in ok], digits=2),
        e2e_tps=Stats.of([r.e2e_tps for r in ok], digits=2),
        latency_s=Stats.of([r.total_s for r in ok], digits=4),
    )


def run_benchmark(
    cfg: BenchConfig,
    prompts: Sequence[str],
    *,
    transport: httpx.BaseTransport | None = None,
    client: httpx.Client | None = None,
    clock: Clock = time.perf_counter,
    on_result: Callable[[RequestResult], None] | None = None,
) -> BenchRun:
    """N промптов × M повторов, последовательно (batch=1 — честная латентность).

    `warmup` первых запросов не учитываются: в Ollama первый запрос загружает модель в VRAM.
    """
    if not prompts:
        raise ValueError("нужен хотя бы один промпт")
    own = client is None
    http = client or httpx.Client(timeout=cfg.timeout, transport=transport)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    results: list[RequestResult] = []
    try:
        for _ in range(cfg.warmup):
            measure_request(http, cfg, prompts[0], clock=clock)
        for rep in range(cfg.repeats):
            for idx, prompt in enumerate(prompts):
                res = measure_request(http, cfg, prompt, clock=clock, prompt_idx=idx, repeat=rep)
                results.append(res)
                if on_result:
                    on_result(res)
    finally:
        if own:
            http.close()
    return BenchRun(
        config=cfg, started_at=started, summary=summarize(cfg, results), results=results
    )


# ---------------------------------------------------------------------------
# Вывод: rich-таблицы, JSON, CSV
# ---------------------------------------------------------------------------


def _fmt(v: float | None, spec: str = ".1f") -> str:
    return "—" if v is None else format(v, spec)


def summary_table(runs: Sequence[BenchRun], title: str = "Бенчмарк") -> Table:
    table = Table(title=title)
    table.add_column("Конфиг", min_width=24)
    for col, justify in [
        ("Запросов", "right"),
        ("Ошибок", "right"),
        ("TTFT p50, мс", "right"),
        ("TTFT p95, мс", "right"),
        ("decode tok/s", "right"),
        ("e2e tok/s", "right"),
        ("p50, с", "right"),
        ("p95, с", "right"),
    ]:
        table.add_column(col, justify=justify)  # type: ignore[arg-type]
    for run in runs:
        s = run.summary
        table.add_row(
            s.label,
            str(s.n_requests),
            str(s.n_errors),
            _fmt(s.ttft_ms.p50),
            _fmt(s.ttft_ms.p95),
            _fmt(s.decode_tps.mean),
            _fmt(s.e2e_tps.mean),
            _fmt(s.latency_s.p50, ".2f"),
            _fmt(s.latency_s.p95, ".2f"),
        )
    return table


# (подпись, извлечение, больше = лучше)
COMPARE_METRICS: list[tuple[str, Callable[[BenchSummary], float | None], bool]] = [
    ("TTFT p50, мс", lambda s: s.ttft_ms.p50, False),
    ("TTFT p95, мс", lambda s: s.ttft_ms.p95, False),
    ("decode tok/s (mean)", lambda s: s.decode_tps.mean, True),
    ("e2e tok/s (mean)", lambda s: s.e2e_tps.mean, True),
    ("Латентность p50, с", lambda s: s.latency_s.p50, False),
    ("Латентность p95, с", lambda s: s.latency_s.p95, False),
    ("Ошибок", lambda s: float(s.n_errors), False),
]


def compare_rows(a: BenchSummary, b: BenchSummary) -> list[dict[str, Any]]:
    rows = []
    for label, get, higher_better in COMPARE_METRICS:
        va, vb = get(a), get(b)
        delta = None
        winner = None
        if va is not None and vb is not None:
            if va:
                delta = round((vb - va) / va * 100, 1)
            if va != vb:
                b_better = (vb > va) if higher_better else (vb < va)
                winner = b.label if b_better else a.label
        rows.append({"metric": label, "a": va, "b": vb, "delta_pct": delta, "winner": winner})
    return rows


def compare_table(a: BenchSummary, b: BenchSummary) -> Table:
    table = Table(title=f"Сравнение: A = {a.label}  vs  B = {b.label}")
    table.add_column("Метрика", min_width=20)
    table.add_column("A", justify="right")
    table.add_column("B", justify="right")
    table.add_column("Δ B к A", justify="right")
    table.add_column("Лучше")
    for row in compare_rows(a, b):
        delta = "—" if row["delta_pct"] is None else f"{row['delta_pct']:+.1f}%"
        table.add_row(
            row["metric"],
            _fmt(row["a"], ".2f"),
            _fmt(row["b"], ".2f"),
            delta,
            row["winner"] or "=",
        )
    return table


def save_json(run: BenchRun, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return p


def load_run(path: str | Path) -> BenchRun:
    return BenchRun.model_validate_json(Path(path).read_text(encoding="utf-8"))


CSV_FIELDS = [
    "label",
    "model",
    "prompt_idx",
    "repeat",
    "ok",
    "ttft_ms",
    "total_s",
    "completion_tokens",
    "prompt_tokens",
    "decode_tps",
    "e2e_tps",
    "tokens_from_usage",
    "error",
]


def save_csv(runs: BenchRun | Sequence[BenchRun], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    items = [runs] if isinstance(runs, BenchRun) else list(runs)
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for run in items:
            for r in run.results:
                writer.writerow(
                    {
                        "label": run.summary.label,
                        "model": run.config.model,
                        "prompt_idx": r.prompt_idx,
                        "repeat": r.repeat,
                        "ok": r.ok,
                        "ttft_ms": None if r.ttft_s is None else round(r.ttft_s * 1000, 3),
                        "total_s": r.total_s,
                        "completion_tokens": r.completion_tokens,
                        "prompt_tokens": r.prompt_tokens,
                        "decode_tps": r.decode_tps,
                        "e2e_tps": r.e2e_tps,
                        "tokens_from_usage": r.tokens_from_usage,
                        "error": r.error,
                    }
                )
    return p
