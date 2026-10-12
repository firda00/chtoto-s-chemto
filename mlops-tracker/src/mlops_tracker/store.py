"""SQLite-хранилище ранов: схема, запись и запросы (list / get / compare / best)."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

DEFAULT_DB = "mlruns/mlt.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    experiment  TEXT NOT NULL DEFAULT 'default',
    status      TEXT NOT NULL DEFAULT 'running',
    start_time  REAL NOT NULL,
    end_time    REAL,
    git_commit  TEXT,
    seed        INTEGER,
    tags        TEXT NOT NULL DEFAULT '[]',
    env         TEXT NOT NULL DEFAULT '{}',
    error       TEXT
);
CREATE TABLE IF NOT EXISTS params (
    run_id  TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    key     TEXT NOT NULL,
    value   TEXT NOT NULL,
    PRIMARY KEY (run_id, key)
);
CREATE TABLE IF NOT EXISTS metrics (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    key       TEXT NOT NULL,
    value     REAL NOT NULL,
    step      INTEGER NOT NULL DEFAULT 0,
    timestamp REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_metrics_run_key ON metrics(run_id, key);
CREATE TABLE IF NOT EXISTS artifacts (
    run_id  TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name    TEXT NOT NULL,
    path    TEXT NOT NULL,
    size    INTEGER,
    PRIMARY KEY (run_id, name)
);
CREATE TABLE IF NOT EXISTS texts (
    run_id  TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    key     TEXT NOT NULL,
    content TEXT NOT NULL,
    PRIMARY KEY (run_id, key)
);
"""


class MetricPoint(BaseModel):
    value: float
    step: int
    timestamp: float


class RunRecord(BaseModel):
    """Полное представление рана, собранное из всех таблиц."""

    id: str
    name: str
    experiment: str
    status: str
    start_time: float
    end_time: float | None = None
    git_commit: str | None = None
    seed: int | None = None
    tags: list[str] = Field(default_factory=list)
    env: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, list[MetricPoint]] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    texts: dict[str, str] = Field(default_factory=dict)

    @property
    def duration_s(self) -> float | None:
        return None if self.end_time is None else self.end_time - self.start_time

    def last_metrics(self) -> dict[str, float]:
        """Последнее (по шагу) значение каждой метрики."""
        return {k: pts[-1].value for k, pts in self.metrics.items() if pts}


class RunNotFoundError(KeyError):
    pass


def default_db_path() -> Path:
    return Path(os.environ.get("MLT_DB") or DEFAULT_DB)


class Store:
    """Тонкая обёртка над sqlite3.

    Соединение открывается на каждую операцию: хранилище безопасно вызывать из разных
    потоков, а файл базы можно спокойно удалить в тестах (на Windows это важно).
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @property
    def artifact_root(self) -> Path:
        return self.db_path.parent / "artifacts"

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ write
    def create_run(
        self,
        run_id: str,
        name: str,
        experiment: str = "default",
        tags: list[str] | None = None,
        git_commit: str | None = None,
        seed: int | None = None,
        env: dict[str, Any] | None = None,
    ) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO runs (id, name, experiment, status, start_time, git_commit, seed,"
                " tags, env) VALUES (?, ?, ?, 'running', ?, ?, ?, ?, ?)",
                (
                    run_id,
                    name,
                    experiment,
                    time.time(),
                    git_commit,
                    seed,
                    json.dumps(tags or [], ensure_ascii=False),
                    json.dumps(env or {}, ensure_ascii=False, default=str),
                ),
            )

    def finish_run(self, run_id: str, status: str = "finished", error: str | None = None) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE runs SET status = ?, end_time = ?, error = ? WHERE id = ?",
                (status, time.time(), error, run_id),
            )

    def set_tags(self, run_id: str, tags: list[str]) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE runs SET tags = ? WHERE id = ?",
                (json.dumps(sorted(set(tags)), ensure_ascii=False), run_id),
            )

    def log_params(self, run_id: str, params: dict[str, Any]) -> None:
        rows = [
            (run_id, k, json.dumps(v, ensure_ascii=False, default=str)) for k, v in params.items()
        ]
        with self._conn() as c:
            c.executemany(
                "INSERT OR REPLACE INTO params (run_id, key, value) VALUES (?, ?, ?)", rows
            )

    def log_metrics(self, run_id: str, metrics: dict[str, float], step: int = 0) -> None:
        now = time.time()
        rows = [(run_id, k, float(v), int(step), now) for k, v in metrics.items()]
        with self._conn() as c:
            c.executemany(
                "INSERT INTO metrics (run_id, key, value, step, timestamp) VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0) -> None:
        self.log_metrics(run_id, {key: value}, step=step)

    def log_artifact(self, run_id: str, name: str, path: str | Path, size: int | None) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO artifacts (run_id, name, path, size) VALUES (?, ?, ?, ?)",
                (run_id, name, str(path), size),
            )

    def log_text(self, run_id: str, key: str, content: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO texts (run_id, key, content) VALUES (?, ?, ?)",
                (run_id, key, content),
            )

    def delete_run(self, run_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM runs WHERE id = ?", (self.resolve_id(run_id),))

    # ------------------------------------------------------------------- read
    def resolve_id(self, run_id_or_prefix: str) -> str:
        """Полный id по префиксу (как короткие хэши в git)."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT id FROM runs WHERE id LIKE ? ORDER BY start_time DESC",
                (run_id_or_prefix + "%",),
            ).fetchall()
        if not rows:
            raise RunNotFoundError(f"Ран не найден: {run_id_or_prefix}")
        ids = [r["id"] for r in rows]
        if run_id_or_prefix in ids:
            return run_id_or_prefix
        if len(ids) > 1:
            raise RunNotFoundError(f"Префикс неоднозначен: {run_id_or_prefix} ({len(ids)} ранов)")
        return ids[0]

    def list_runs(
        self,
        experiment: str | None = None,
        tag: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[RunRecord]:
        sql = "SELECT id FROM runs"
        cond: list[str] = []
        args: list[Any] = []
        if experiment:
            cond.append("experiment = ?")
            args.append(experiment)
        if status:
            cond.append("status = ?")
            args.append(status)
        if cond:
            sql += " WHERE " + " AND ".join(cond)
        sql += " ORDER BY start_time DESC, rowid DESC"
        with self._conn() as c:
            ids = [r["id"] for r in c.execute(sql, args).fetchall()]
        runs = [self.get_run(i) for i in ids]
        if tag:
            runs = [r for r in runs if tag in r.tags]
        return runs[:limit] if limit else runs

    def get_run(self, run_id: str) -> RunRecord:
        run_id = self.resolve_id(run_id)
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            params = {
                r["key"]: json.loads(r["value"])
                for r in c.execute("SELECT key, value FROM params WHERE run_id = ?", (run_id,))
            }
            metrics: dict[str, list[MetricPoint]] = {}
            for r in c.execute(
                "SELECT key, value, step, timestamp FROM metrics WHERE run_id = ?"
                " ORDER BY key, step, id",
                (run_id,),
            ):
                metrics.setdefault(r["key"], []).append(
                    MetricPoint(value=r["value"], step=r["step"], timestamp=r["timestamp"])
                )
            artifacts = {
                r["name"]: r["path"]
                for r in c.execute("SELECT name, path FROM artifacts WHERE run_id = ?", (run_id,))
            }
            texts = {
                r["key"]: r["content"]
                for r in c.execute("SELECT key, content FROM texts WHERE run_id = ?", (run_id,))
            }
        return RunRecord(
            id=row["id"],
            name=row["name"],
            experiment=row["experiment"],
            status=row["status"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            git_commit=row["git_commit"],
            seed=row["seed"],
            tags=json.loads(row["tags"]),
            env=json.loads(row["env"]),
            error=row["error"],
            params=params,
            metrics=metrics,
            artifacts=artifacts,
            texts=texts,
        )

    def get_text(self, run_id: str, key: str) -> str | None:
        run_id = self.resolve_id(run_id)
        with self._conn() as c:
            row = c.execute(
                "SELECT content FROM texts WHERE run_id = ? AND key = ?", (run_id, key)
            ).fetchone()
        return row["content"] if row else None

    def compare_runs(self, run_a: str, run_b: str) -> dict[str, Any]:
        """Сравнение двух ранов: параметры (что поменялось) и последние метрики с дельтой."""
        a, b = self.get_run(run_a), self.get_run(run_b)
        params = {}
        for k in sorted(set(a.params) | set(b.params)):
            va, vb = a.params.get(k), b.params.get(k)
            params[k] = {"a": va, "b": vb, "changed": va != vb}
        ma, mb = a.last_metrics(), b.last_metrics()
        metrics = {}
        for k in sorted(set(ma) | set(mb)):
            va, vb = ma.get(k), mb.get(k)
            delta = vb - va if va is not None and vb is not None else None
            metrics[k] = {"a": va, "b": vb, "delta": delta}
        return {
            "a": {"id": a.id, "name": a.name, "git_commit": a.git_commit, "seed": a.seed},
            "b": {"id": b.id, "name": b.name, "git_commit": b.git_commit, "seed": b.seed},
            "params": params,
            "metrics": metrics,
        }

    def best_run(
        self,
        metric: str,
        mode: Literal["max", "min"] = "max",
        experiment: str | None = None,
        tag: str | None = None,
    ) -> tuple[RunRecord, float] | None:
        """Лучший завершённый ран по последнему значению метрики (max или min)."""
        if mode not in ("max", "min"):
            raise ValueError("mode должен быть 'max' или 'min'")
        sql = """
            SELECT m.run_id, m.value FROM (
                SELECT run_id, value,
                       ROW_NUMBER() OVER (PARTITION BY run_id ORDER BY step DESC, id DESC) AS rn
                FROM metrics WHERE key = ?
            ) AS m JOIN runs r ON r.id = m.run_id
            WHERE m.rn = 1 AND r.status = 'finished'
        """
        args: list[Any] = [metric]
        if experiment:
            sql += " AND r.experiment = ?"
            args.append(experiment)
        sql += f" ORDER BY m.value {'DESC' if mode == 'max' else 'ASC'}, r.start_time DESC"
        with self._conn() as c:
            rows = c.execute(sql, args).fetchall()
        for row in rows:
            run = self.get_run(row["run_id"])
            if tag is None or tag in run.tags:
                return run, row["value"]
        return None
