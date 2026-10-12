"""CLI: `uv run llm <команда>`."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from local_llm import bench as bench_mod
from local_llm.backends import ollama as ollama_mod
from local_llm.backends import vllm as vllm_mod
from local_llm.client import OLLAMA_OPENAI_URL, get_llm
from local_llm.mock_server import FakeClock, MockProfile, make_mock_transport, simulated_profile
from local_llm.presets import (
    QUANTS,
    VRAM_TIERS,
    Backend,
    ModelPreset,
    get_preset,
    get_quant,
    load_presets,
    memory_breakdown,
)
from local_llm.tune import TARGETS, Candidate, Target, recommend

app = typer.Typer(
    help="Запуск, бенчмарк и оптимизация локальных LLM (Qwen, Llama) через Ollama и vLLM.",
    no_args_is_help=True,
    add_completion=False,
)
ollama_app = typer.Typer(help="Нативный API Ollama: список, загрузка, информация о моделях.")
app.add_typer(ollama_app, name="ollama")


def console() -> Console:
    return Console(width=None if sys.stdout.isatty() else 140, highlight=False)


@app.callback()
def _main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure and (getattr(stream, "encoding", "") or "").lower() != "utf-8":
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


def _preset(preset_id: str) -> ModelPreset:
    try:
        return get_preset(preset_id)
    except KeyError as exc:
        raise typer.BadParameter(str(exc.args[0])) from None


def _check_backend(backend: str) -> Backend:
    if backend not in ("ollama", "vllm"):
        raise typer.BadParameter("backend: ollama или vllm")
    return backend  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# presets / estimate / recommend
# ---------------------------------------------------------------------------


@app.command()
def presets(
    backend: Annotated[str, typer.Option(help="ollama | vllm")] = "ollama",
    as_json: Annotated[bool, typer.Option("--json", help="Вывести каталог в JSON")] = False,
) -> None:
    """Каталог пресетов и рекомендованные квантования по VRAM (8/12/16/24 GB)."""
    be = _check_backend(backend)
    catalog = load_presets()
    if as_json:
        typer.echo(
            json.dumps(
                {k: v.model_dump(mode="json") for k, v in catalog.items()},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    table = Table(title=f"Пресеты моделей · бэкенд {be}")
    table.add_column("id")
    table.add_column("Модель")
    table.add_column("Параметры", justify="right")
    table.add_column("max ctx", justify="right")
    table.add_column("Квантования")
    for tier in VRAM_TIERS:
        table.add_column(f"{tier} GB", justify="center")
    for p in catalog.values():
        recs = p.recommended.get(be, {})
        cells = []
        for tier in VRAM_TIERS:
            rec = recs.get(tier)
            cells.append("—" if rec is None else f"{rec.quant}\n{rec.ctx // 1024}k ctx")
        table.add_row(
            p.id,
            p.display_name,
            f"{p.params_b:g}B",
            f"{p.max_ctx // 1024}k",
            ", ".join(p.quants(be)),
            *cells,
        )
    out = console()
    out.print(table)
    qt = Table(title="Квантования")
    qt.add_column("имя")
    qt.add_column("бэкенды")
    qt.add_column("бит/вес", justify="right")
    qt.add_column("качество", justify="right")
    qt.add_column("описание")
    for q in QUANTS.values():
        qt.add_row(q.name, ", ".join(q.backends), f"{q.bits:g}", f"{q.quality:.3f}", q.description)
    out.print(qt)


@app.command()
def estimate(
    model: Annotated[str | None, typer.Option("--model", "-m", help="id пресета")] = None,
    quant: Annotated[str, typer.Option(help="q4_K_M | q5_K_M | q8_0 | fp16 | awq | gptq | fp8")] = (
        "q4_K_M"
    ),
    ctx: Annotated[int, typer.Option(help="Длина контекста, токенов")] = 8192,
    batch: Annotated[int, typer.Option(help="Одновременных последовательностей")] = 1,
    kv_bits: Annotated[float, typer.Option(help="Точность KV-кэша: 16 или 8 (fp8/q8_0)")] = 16,
    params: Annotated[float | None, typer.Option(help="Своя модель: параметров, млрд")] = None,
    layers: Annotated[int, typer.Option(help="Своя модель: число слоёв")] = 32,
    kv_heads: Annotated[int, typer.Option(help="Своя модель: KV-голов (GQA)")] = 8,
    head_dim: Annotated[int, typer.Option(help="Своя модель: размер головы")] = 128,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Оценка VRAM: веса × бит/8 × overhead + KV-кэш + runtime."""
    try:
        q = get_quant(quant)
    except KeyError as exc:
        raise typer.BadParameter(str(exc.args[0])) from None
    if model:
        p = _preset(model)
        est = p.estimate(quant, ctx, kv_bits=kv_bits, batch=batch)
        title = f"{p.display_name} · {quant} · ctx {ctx}"
    elif params:
        est = memory_breakdown(
            params,
            q.bits,
            ctx,
            n_layers=layers,
            n_kv_heads=kv_heads,
            head_dim=head_dim,
            kv_bits=kv_bits,
            batch=batch,
        )
        title = f"{params:g}B · {quant} · ctx {ctx}"
    else:
        raise typer.BadParameter("Укажите --model <preset> или --params <млрд>")
    fits = {tier: est.fits(tier, headroom=0.1) for tier in VRAM_TIERS}
    if as_json:
        typer.echo(json.dumps({"title": title, **est.model_dump(), "fits_90pct": fits}, indent=2))
        return
    table = Table(title=f"Оценка памяти: {title}")
    table.add_column("Компонент")
    table.add_column("GiB", justify="right")
    table.add_row("Веса (× overhead 1.10)", f"{est.weights_gb:.2f}")
    table.add_row(f"KV-кэш ({kv_bits:g} бит, batch {batch})", f"{est.kv_cache_gb:.2f}")
    table.add_row("Runtime (CUDA-контекст, буферы)", f"{est.runtime_gb:.2f}")
    table.add_row("[bold]Итого[/bold]", f"[bold]{est.total_gb:.2f}[/bold]")
    out = console()
    out.print(table)
    verdict = "  ".join(f"{t} GB: {'✓' if ok else '✗'}" for t, ok in fits.items())
    out.print(f"Помещается (90% VRAM): {verdict}")


def _candidate_table(title: str, rows: list[Candidate]) -> Table:
    table = Table(title=title)
    table.add_column("#", justify="right")
    table.add_column("Пресет")
    table.add_column("Квант")
    table.add_column("ctx", justify="right")
    table.add_column("VRAM, GiB", justify="right")
    table.add_column("Качество", justify="right")
    table.add_column("Скорость*", justify="right")
    table.add_column("Score", justify="right")
    for i, c in enumerate(rows, 1):
        table.add_row(
            str(i),
            c.preset_id,
            c.quant,
            str(c.ctx),
            f"{c.memory.total_gb:.2f}",
            f"{c.quality:.3f}",
            f"{c.speed:.3f}",
            f"{c.score:.3f}",
        )
    return table


@app.command(name="recommend")
def recommend_cmd(
    vram: Annotated[float, typer.Option(help="Бюджет VRAM, GB")],
    target: Annotated[str, typer.Option(help="latency | balanced | quality")] = "balanced",
    backend: Annotated[str, typer.Option(help="ollama | vllm")] = "ollama",
    min_ctx: Annotated[int, typer.Option(help="Минимальный контекст")] = 4096,
    max_ctx: Annotated[int | None, typer.Option(help="Потолок контекста (иначе по цели)")] = None,
    headroom: Annotated[float, typer.Option(help="Доля VRAM в запасе")] = 0.10,
    top: Annotated[int, typer.Option(help="Сколько вариантов показать")] = 5,
    as_json: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Подобрать пресет + квантование + контекст под VRAM и цель."""
    if target not in TARGETS:
        raise typer.BadParameter(f"target: {' | '.join(TARGETS)}")
    be = _check_backend(backend)
    tgt: Target = target  # type: ignore[assignment]
    rec = recommend(
        vram,
        target=tgt,
        backend=be,
        min_ctx=min_ctx,
        max_ctx=max_ctx,
        headroom=headroom,
        top=top,
    )
    if as_json:
        typer.echo(rec.model_dump_json(indent=2))
        return
    out = console()
    if rec.best is None:
        out.print(
            f"[red]Ни одна конфигурация не помещается в {vram:g} GB "
            f"(доступно {rec.usable_gb:g} GiB) с контекстом ≥ {min_ctx}.[/red]\n"
            "Попробуйте: --min-ctx 2048, бэкенд ollama с частичным offload (num_gpu), "
            "или KV-кэш в 8 бит."
        )
        raise typer.Exit(1)
    b = rec.best
    out.print(
        f"[bold green]Рекомендация[/bold green] для {vram:g} GB · цель {tgt} · {be}: "
        f"[bold]{b.display_name}[/bold] · {b.quant} · ctx {b.ctx} · "
        f"~{b.memory.total_gb:.1f} GiB из {rec.usable_gb:g}"
    )
    out.print(_candidate_table("Варианты", [b, *rec.alternatives]))
    out.print("* скорость ≈ 1/размер весов (декодинг упирается в пропускную способность памяти)")
    if be == "ollama":
        out.print("\nСледующие шаги:")
        out.print(f"  ollama pull {b.ollama_tag}")
        out.print(f"  uv run llm modelfile --preset {b.preset_id} --quant {b.quant} --ctx {b.ctx}")
    else:
        out.print("\nСледующий шаг:")
        out.print(
            f"  uv run llm vllm-cmd --preset {b.preset_id} --quant {b.quant} "
            f"--max-model-len {b.ctx}"
        )


# ---------------------------------------------------------------------------
# modelfile / vllm-cmd
# ---------------------------------------------------------------------------


@app.command()
def modelfile(
    preset: Annotated[str, typer.Option(help="id пресета")],
    quant: Annotated[str, typer.Option(help="q4_K_M | q5_K_M | q8_0 | fp16")] = "q4_K_M",
    ctx: Annotated[int | None, typer.Option(help="num_ctx")] = None,
    num_gpu: Annotated[int | None, typer.Option(help="Слоёв на GPU (99 = все)")] = None,
    temperature: Annotated[float | None, typer.Option()] = None,
    top_p: Annotated[float | None, typer.Option()] = None,
    repeat_penalty: Annotated[float | None, typer.Option()] = None,
    system: Annotated[str | None, typer.Option(help="Системный промпт")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Записать в файл")] = None,
) -> None:
    """Сгенерировать Ollama Modelfile из пресета."""
    p = _preset(preset)
    opts = ollama_mod.options_from_preset(
        p,
        num_ctx=ctx,
        num_gpu=num_gpu,
        temperature=temperature,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
    )
    try:
        text = ollama_mod.render_modelfile(p, quant, opts, system=system)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    name = ollama_mod.custom_model_name(p, quant, opts.num_ctx)
    if output:
        output.write_text(text, encoding="utf-8")
        typer.echo(f"Записано: {output}")
        typer.echo(f"Создать модель: ollama create {name} -f {output}")
    else:
        typer.echo(text, nl=False)
        typer.echo(f"# ollama create {name} -f Modelfile", err=True)


@app.command(name="vllm-cmd")
def vllm_cmd(
    preset: Annotated[str, typer.Option(help="id пресета")],
    quant: Annotated[str, typer.Option(help="awq | gptq | fp8 | fp16")] = "awq",
    max_model_len: Annotated[int | None, typer.Option(help="--max-model-len")] = None,
    gpu_mem: Annotated[float, typer.Option(help="--gpu-memory-utilization")] = 0.90,
    tp: Annotated[int, typer.Option(help="--tensor-parallel-size")] = 1,
    kv_cache_dtype: Annotated[str, typer.Option(help="auto | fp8")] = "auto",
    max_num_seqs: Annotated[int | None, typer.Option(help="--max-num-seqs")] = None,
    port: Annotated[int, typer.Option()] = 8000,
    compose: Annotated[bool, typer.Option("--compose", help="Вывести docker-compose")] = False,
    one_line: Annotated[bool, typer.Option("--one-line", help="Команда одной строкой")] = False,
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Сгенерировать команду `vllm serve` (или docker-compose) из пресета."""
    p = _preset(preset)
    if kv_cache_dtype not in ("auto", "fp8"):
        raise typer.BadParameter("kv-cache-dtype: auto | fp8")
    try:
        cfg = vllm_mod.config_from_preset(
            p,
            quant,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_mem,
            tensor_parallel_size=tp,
            kv_cache_dtype=kv_cache_dtype,  # type: ignore[arg-type]
            max_num_seqs=max_num_seqs,
            port=port,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    if compose:
        text = vllm_mod.render_compose(cfg)
    else:
        text = vllm_mod.render_shell(cfg, multiline=not one_line) + "\n"
    if output:
        output.write_text(text, encoding="utf-8")
        typer.echo(f"Записано: {output}")
    else:
        typer.echo(text, nl=False)
    est = p.estimate(quant, cfg.max_model_len, kv_bits=8 if kv_cache_dtype == "fp8" else 16)
    typer.echo(
        f"# Оценка: {est.total_gb:.1f} GiB (веса {est.weights_gb:.1f}, KV {est.kv_cache_gb:.1f}); "
        f"vLLM займёт {gpu_mem:.0%} VRAM, остаток уйдёт под KV-кэш батча",
        err=True,
    )


# ---------------------------------------------------------------------------
# bench / compare
# ---------------------------------------------------------------------------


def _bench_one(
    cfg: bench_mod.BenchConfig,
    prompts: list[str],
    *,
    mock: bool,
    mock_profile: MockProfile,
    quiet: bool,
) -> bench_mod.BenchRun:
    out = console()

    def show(r: bench_mod.RequestResult) -> None:
        if quiet:
            return
        if r.ok:
            out.print(
                f"  [{cfg.name}] prompt {r.prompt_idx} rep {r.repeat}: "
                f"TTFT {r.ttft_s * 1000 if r.ttft_s else 0:.0f} мс, "
                f"{r.decode_tps or 0:.1f} tok/s, {r.completion_tokens} ток., {r.total_s:.2f} с"
            )
        else:
            out.print(f"  [red][{cfg.name}] prompt {r.prompt_idx}: {r.error}[/red]")

    if mock:
        clock = FakeClock()
        transport = make_mock_transport(
            clock=clock,
            profile_for=lambda m: simulated_profile(m, mock_profile),
            vary_length=True,
        )
        return bench_mod.run_benchmark(
            cfg, prompts, transport=transport, clock=clock, on_result=show
        )
    return bench_mod.run_benchmark(cfg, prompts, on_result=show)


@app.command()
def bench(
    model: Annotated[str | None, typer.Option(help="Модель (по умолчанию $LLM_MODEL)")] = None,
    base_url: Annotated[
        str | None, typer.Option(help="OpenAI-совместимый URL (по умолчанию $LLM_BASE_URL)")
    ] = None,
    prompts: Annotated[Path, typer.Option(help="Файл с промптами")] = Path("data/prompts.txt"),
    repeats: Annotated[int, typer.Option(help="Повторов каждого промпта")] = 1,
    max_tokens: Annotated[int, typer.Option()] = 256,
    temperature: Annotated[float, typer.Option()] = 0.0,
    warmup: Annotated[int, typer.Option(help="Прогревочных запросов (не учитываются)")] = 1,
    limit: Annotated[int | None, typer.Option(help="Взять первые N промптов")] = None,
    label: Annotated[str | None, typer.Option()] = None,
    api_key: Annotated[str | None, typer.Option(help="По умолчанию $LLM_API_KEY")] = None,
    model_b: Annotated[str | None, typer.Option(help="Вторая модель для сравнения")] = None,
    base_url_b: Annotated[str | None, typer.Option(help="URL второго конфига")] = None,
    label_b: Annotated[str | None, typer.Option()] = None,
    json_out: Annotated[Path | None, typer.Option("--json", help="Сохранить JSON")] = None,
    csv_out: Annotated[Path | None, typer.Option("--csv", help="Сохранить CSV")] = None,
    mock: Annotated[
        bool, typer.Option("--mock", help="Фейковый in-process сервер (без Ollama/vLLM)")
    ] = False,
    mock_ttft_ms: Annotated[float, typer.Option(help="--mock: TTFT для q4_K_M, мс")] = 180.0,
    mock_tps: Annotated[float, typer.Option(help="--mock: tok/s для q4_K_M")] = 45.0,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """Бенчмарк TTFT / tok/s / латентности по N промптов × M повторов (+ сравнение A/B)."""
    model = model or os.environ.get("LLM_MODEL")
    if not model:
        raise typer.BadParameter("Укажите --model или LLM_MODEL")
    base_url = base_url or os.environ.get("LLM_BASE_URL") or OLLAMA_OPENAI_URL
    key = api_key or os.environ.get("LLM_API_KEY") or "local"
    try:
        prompt_list = bench_mod.load_prompts(prompts)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"prompts: {exc}") from None
    if limit:
        prompt_list = prompt_list[:limit]
    common = {
        "api_key": key,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "repeats": repeats,
        "warmup": warmup,
    }
    configs = [bench_mod.BenchConfig(base_url=base_url, model=model, label=label, **common)]
    if model_b or base_url_b:
        configs.append(
            bench_mod.BenchConfig(
                base_url=base_url_b or base_url,
                model=model_b or model,
                label=label_b or (f"{model_b or model} @ {base_url_b}" if base_url_b else None),
                **common,
            )
        )
    if len(configs) == 2 and configs[0].name == configs[1].name:
        configs[0].label, configs[1].label = f"A: {configs[0].name}", f"B: {configs[1].name}"

    out = console()
    profile = MockProfile(ttft_s=mock_ttft_ms / 1000, tps=mock_tps, max_tokens=max_tokens)
    runs = []
    for cfg in configs:
        out.print(
            f"[bold]▶ {cfg.name}[/bold] · {cfg.base_url} · {len(prompt_list)} промптов × "
            f"{repeats} · warmup {warmup}{' · MOCK' if mock else ''}"
        )
        runs.append(_bench_one(cfg, prompt_list, mock=mock, mock_profile=profile, quiet=quiet))

    out.print(bench_mod.summary_table(runs))
    if len(runs) == 2:
        out.print(bench_mod.compare_table(runs[0].summary, runs[1].summary))
    if json_out:
        if len(runs) == 1:
            bench_mod.save_json(runs[0], json_out)
            out.print(f"JSON: {json_out}")
        else:
            for run, suffix in zip(runs, ("a", "b"), strict=True):
                path = json_out.with_name(f"{json_out.stem}_{suffix}{json_out.suffix}")
                bench_mod.save_json(run, path)
                out.print(f"JSON: {path}")
    if csv_out:
        bench_mod.save_csv(runs, csv_out)
        out.print(f"CSV: {csv_out}")
    if any(r.summary.n_errors == r.summary.n_requests for r in runs):
        raise typer.Exit(1)


@app.command()
def compare(
    run_a: Annotated[Path, typer.Argument(help="JSON прогона A")],
    run_b: Annotated[Path, typer.Argument(help="JSON прогона B")],
) -> None:
    """Сравнить два сохранённых прогона бенчмарка бок о бок."""
    a, b = bench_mod.load_run(run_a), bench_mod.load_run(run_b)
    out = console()
    out.print(bench_mod.summary_table([a, b]))
    out.print(bench_mod.compare_table(a.summary, b.summary))


# ---------------------------------------------------------------------------
# ask + ollama
# ---------------------------------------------------------------------------


@app.command()
def ask(
    prompt: Annotated[str, typer.Argument()],
    system: Annotated[str | None, typer.Option()] = None,
    model: Annotated[str | None, typer.Option(help="По умолчанию $LLM_MODEL")] = None,
    base_url: Annotated[str | None, typer.Option(help="По умолчанию $LLM_BASE_URL")] = None,
    mock: Annotated[bool, typer.Option("--mock", help="MockLLM вместо сети")] = False,
) -> None:
    """Один запрос через единый клиент (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL)."""
    llm = get_llm(mock=mock or None, model=model, base_url=base_url)
    typer.echo(llm.complete(prompt, system=system))


HostOpt = Annotated[str | None, typer.Option(help="По умолчанию $OLLAMA_HOST или :11434")]


def _ollama_call(host: str | None, fn):
    import httpx

    with ollama_mod.OllamaClient(host) as cli:
        try:
            return fn(cli)
        except httpx.HTTPError as exc:
            typer.echo(
                f"Ollama недоступна ({cli.host}): {exc}. Запущен ли `ollama serve`?", err=True
            )
            raise typer.Exit(2) from None


@ollama_app.command("list")
def ollama_list(host: HostOpt = None) -> None:
    """Локальные модели (/api/tags)."""
    models = _ollama_call(host, lambda c: c.list_models())
    table = Table(title=f"Ollama · {len(models)} моделей")
    table.add_column("Модель")
    table.add_column("Размер, GB", justify="right")
    for m in models:
        table.add_row(m.get("name", "?"), f"{(m.get('size') or 0) / 1e9:.1f}")
    console().print(table)


@ollama_app.command("pull")
def ollama_pull(tag: Annotated[str, typer.Argument()], host: HostOpt = None) -> None:
    """Скачать модель (/api/pull)."""
    typer.echo(json.dumps(_ollama_call(host, lambda c: c.pull(tag)), ensure_ascii=False))


@ollama_app.command("show")
def ollama_show(tag: Annotated[str, typer.Argument()], host: HostOpt = None) -> None:
    """Параметры и квантование модели (/api/show)."""
    info = _ollama_call(host, lambda c: c.show(tag))
    typer.echo(json.dumps({k: info.get(k) for k in ("details", "parameters")}, indent=2))


if __name__ == "__main__":
    app()
