"""CLI: `rag init-db | ingest | ask | eval | serve`."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
from pathlib import Path

import typer

from rag_system.chunking import get_chunker
from rag_system.config import Settings, load_dotenv
from rag_system.factory import Runtime, evaluate, load_eval
from rag_system.llm import resolve_provider

app = typer.Typer(
    help="Naive RAG и Modular RAG: ingest → ask → eval.", no_args_is_help=True, add_completion=False
)


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            with contextlib.suppress(ValueError, OSError):
                stream.reconfigure(encoding="utf-8", errors="replace")


@app.callback()
def main(
    mock: bool = typer.Option(False, "--mock", help="LLM_PROVIDER=mock: без сети и ключей"),
    store: str | None = typer.Option(None, help="memory | pgvector (RAG_STORE)"),
    embedder: str | None = typer.Option(None, help="auto | fastembed | hashing (EMBEDDER)"),
    index_dir: str | None = typer.Option(None, help="папка in-memory индекса (RAG_INDEX_DIR)"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    _utf8_stdout()
    load_dotenv()
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")
    if mock:
        os.environ["LLM_PROVIDER"] = "mock"
    if store:
        os.environ["RAG_STORE"] = store
    if embedder:
        os.environ["EMBEDDER"] = embedder
    if index_dir:
        os.environ["RAG_INDEX_DIR"] = index_dir


def _settings(mock: bool = False) -> Settings:
    if mock:
        os.environ["LLM_PROVIDER"] = "mock"
    return Settings.from_env()


@app.command("init-db")
def init_db() -> None:
    """Миграция pgvector: расширение, таблица, HNSW/IVFFlat-индекс (RAG_STORE=pgvector)."""
    s = _settings()
    s.store = "pgvector"
    rt = Runtime.open(s, fresh=True)
    rt.init_db()
    typer.echo(f"pgvector готов: таблица {s.pg_table}, dim={rt.embedder.dim}, индекс {s.pg_index}")


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, help="файл или папка с .md/.txt"),
    chunker: str = typer.Option("markdown", help="markdown | fixed"),
    chunk_size: int = typer.Option(800, help="макс. размер фрагмента, символов"),
    overlap: int = typer.Option(80, help="перекрытие, символов"),
    append: bool = typer.Option(False, "--append", help="не очищать индекс перед загрузкой"),
) -> None:
    """Разбить документы на фрагменты, посчитать эмбеддинги и сохранить в хранилище."""
    s = _settings()
    rt = Runtime.open(s, fresh=not append)
    stats = rt.ingest(path, get_chunker(chunker, chunk_size, overlap), reset=not append)
    where = s.index_dir if s.store == "memory" else s.pg_dsn.rsplit("@", 1)[-1]
    typer.echo(
        f"Загружено: {stats.pop('documents')} документов → {stats.pop('chunks')} фрагментов "
        f"[{rt.embedder.name}, dim={rt.embedder.dim}] в {s.store} ({where})"
    )
    typer.echo("Коллекции: " + ", ".join(f"{k}={v}" for k, v in sorted(stats.items())))


@app.command()
def ask(
    question: str = typer.Argument(..., help="вопрос"),
    mode: str = typer.Option("modular", "--mode", "-m", help="naive | modular"),
    k: int | None = typer.Option(None, "--k", help="сколько фрагментов в контекст"),
    mock: bool = typer.Option(False, "--mock", help="использовать MockLLM"),
    as_json: bool = typer.Option(False, "--json", help="вывести ответ как JSON"),
) -> None:
    """Задать вопрос базе знаний."""
    s = _settings(mock)
    rt = Runtime.open(s)
    if rt.store.count() == 0:
        typer.echo("Индекс пуст. Сначала: uv run rag ingest data/docs", err=True)
        raise typer.Exit(1)
    res = rt.pipeline(mode, k).answer(question)
    if as_json:
        typer.echo(json.dumps(res.model_dump(), ensure_ascii=False, indent=2))
        return
    typer.echo(
        f"[{res.mode} | llm={rt.llm.name} | embedder={rt.embedder.name} | "
        f"reranker={rt.reranker.name}]"
    )
    if res.rewritten_query is not None:
        typer.echo(f"rewritten: {res.rewritten_query}")
    if res.route is not None:
        r = res.route
        typer.echo(
            f"route: collection={r.get('collection') or 'all'} tools={r.get('tools')} "
            f"({r.get('router')})"
        )
    for tc in res.tool_calls:
        typer.echo(
            f"tool: {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)}) → {tc.result[:120]}"
        )
    typer.echo(f"\nОтвет: {res.answer}\n")
    typer.echo("Источники:")
    for c in res.citations:
        head = f" — {c.heading}" if c.heading else ""
        typer.echo(f"  [{c.n}] {c.source}{head}")


@app.command("eval")
def eval_cmd(
    data: Path = typer.Option(Path("data/eval.jsonl"), exists=True, help="JSONL с вопросами"),
    mode: str = typer.Option("both", "--mode", "-m", help="naive | modular | both"),
    k: int | None = typer.Option(None, "--k"),
    mock: bool = typer.Option(False, "--mock", help="использовать MockLLM"),
    details: bool = typer.Option(False, "--details", help="печатать каждую строку"),
) -> None:
    """Посчитать hit-rate@k и answer-contains accuracy на небольшом QA-наборе."""
    s = _settings(mock)
    rt = Runtime.open(s)
    if rt.store.count() == 0:
        typer.echo("Индекс пуст. Сначала: uv run rag ingest data/docs", err=True)
        raise typer.Exit(1)
    items = load_eval(data)
    k = k or s.top_k
    modes = ["naive", "modular"] if mode == "both" else [mode]
    typer.echo(
        f"eval: {len(items)} вопросов, k={k}, llm={resolve_provider(s.llm_provider)}, "
        f"embedder={rt.embedder.name}, reranker={rt.reranker.name}\n"
    )
    typer.echo(f"{'mode':<10}{'hit@' + str(k):>10}{'answer':>10}")
    for m in modes:
        rep = evaluate(rt.pipeline(m, k), items, k)
        typer.echo(f"{m:<10}{rep['hit_rate']:>10.2f}{rep['answer_accuracy']:>10.2f}")
        if details:
            for row in rep["rows"]:
                mark = ("H" if row["hit"] else "-") + ("A" if row["contains"] else "-")
                typer.echo(f"   {mark} {row['question']} → {row['answer'][:90]}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
) -> None:
    """Запустить HTTP API (FastAPI): POST /ask, POST /ingest, GET /health."""
    import uvicorn

    from rag_system.api import create_app

    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    app()
