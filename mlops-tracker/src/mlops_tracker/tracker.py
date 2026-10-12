"""Трекер экспериментов: `Run` как контекстный менеджер + снапшот окружения для воспроизводимости.

tracker = Tracker("mlruns/mlt.db")
with tracker.start_run("baseline", params={"lr": 1e-3}, tags=["qa"], seed=42) as run:
    run.log_metric("accuracy", 0.9, step=1)
    run.log_artifact("config.yaml")
    run.log_text("prompt", "Ты — помощник ...")
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import shutil
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from types import TracebackType
from typing import Any

from mlops_tracker.store import Store

# Пакеты, версии которых обязательно попадают в снапшот (остальные — если установлены).
KEY_PACKAGES = ("mlops-tracker", "pydantic", "typer", "rich", "jinja2", "openai", "pyyaml", "numpy")


def git_info(cwd: str | Path | None = None) -> dict[str, Any]:
    """Хэш коммита и флаг «грязного» дерева. Любая ошибка (нет git, не репозиторий,
    нет коммитов, таймаут) — не фатальна: возвращаем пустой dict."""
    if os.environ.get("MLT_DISABLE_GIT") == "1":
        return {}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def package_versions(all_packages: bool = True) -> dict[str, str]:
    versions: dict[str, str] = {}
    if all_packages:
        for dist in importlib.metadata.distributions():
            name = dist.metadata.get("Name")
            if name:
                versions[name.lower()] = dist.version
    for name in KEY_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return dict(sorted(versions.items()))


def environment_snapshot() -> dict[str, Any]:
    """Всё, что нужно, чтобы воспроизвести ран: интерпретатор, ОС, версии пакетов, команда."""
    return {
        "python": sys.version.split()[0],
        "python_impl": platform.python_implementation(),
        "platform": platform.platform(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "packages": package_versions(),
    }


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # для дочерних процессов
    try:  # numpy не обязателен
        import numpy as np  # type: ignore[import-not-found]

        np.random.seed(seed)
    except ImportError:
        pass


class Run:
    """Один эксперимент. Используется как контекстный менеджер: при выходе статус
    становится `finished`, а при исключении — `failed` с сохранённым traceback."""

    def __init__(
        self,
        store: Store,
        name: str,
        params: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        experiment: str = "default",
        seed: int | None = None,
        capture_env: bool = True,
    ) -> None:
        self.store = store
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.experiment = experiment
        self.seed = seed if seed is not None else random.SystemRandom().randrange(2**31)
        self.tags = list(tags or [])
        self._finished = False

        git = git_info() if capture_env else {}
        env = environment_snapshot() if capture_env else {}
        if git:
            env["git"] = git
        set_global_seed(self.seed)
        store.create_run(
            self.id,
            name,
            experiment=experiment,
            tags=self.tags,
            git_commit=git.get("commit"),
            seed=self.seed,
            env=env,
        )
        if params:
            self.log_params(params)

    # --------------------------------------------------------------- logging
    @property
    def artifact_dir(self) -> Path:
        return self.store.artifact_root / self.id

    def log_param(self, key: str, value: Any) -> None:
        self.store.log_params(self.id, {key: value})

    def log_params(self, params: dict[str, Any]) -> None:
        self.store.log_params(self.id, params)

    def log_metric(self, key: str, value: float, step: int = 0) -> None:
        self.store.log_metric(self.id, key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        self.store.log_metrics(self.id, metrics, step=step)

    def log_artifact(self, path: str | Path, name: str | None = None) -> Path:
        """Копирует файл в папку артефактов рана и регистрирует его в базе."""
        src = Path(path)
        if not src.is_file():
            raise FileNotFoundError(src)
        name = name or src.name
        dst = self.artifact_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        self.store.log_artifact(self.id, name, dst, dst.stat().st_size)
        return dst

    def log_text(self, key: str, text: str, as_file: bool = False) -> None:
        """Сохраняет текст (промпт, конфиг, ответ) прямо в SQLite; опционально и файлом."""
        self.store.log_text(self.id, key, text)
        if as_file:
            dst = self.artifact_dir / f"{key}.txt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(text, encoding="utf-8")
            self.store.log_artifact(self.id, f"{key}.txt", dst, dst.stat().st_size)

    def set_tag(self, tag: str) -> None:
        self.tags.append(tag)
        self.store.set_tags(self.id, self.tags)

    def end(self, status: str = "finished", error: str | None = None) -> None:
        if not self._finished:
            self.store.finish_run(self.id, status=status, error=error)
            self._finished = True

    # -------------------------------------------------------- context manager
    def __enter__(self) -> Run:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.end("finished")
        else:
            self.end("failed", error="".join(traceback.format_exception(exc_type, exc, tb)))

    def __repr__(self) -> str:
        return f"Run(id={self.id!r}, name={self.name!r}, experiment={self.experiment!r})"


class Tracker:
    def __init__(self, db_path: str | Path | None = None, store: Store | None = None) -> None:
        self.store = store or Store(db_path)

    def start_run(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        experiment: str = "default",
        seed: int | None = None,
        capture_env: bool = True,
    ) -> Run:
        return Run(
            self.store,
            name,
            params=params,
            tags=tags,
            experiment=experiment,
            seed=seed,
            capture_env=capture_env,
        )


_default: Tracker | None = None


def get_tracker() -> Tracker:
    """Трекер по умолчанию (база из `MLT_DB` или `mlruns/mlt.db`)."""
    global _default
    if _default is None:
        _default = Tracker()
    return _default


def start_run(name: str, **kwargs: Any) -> Run:
    """Шорткат: `with start_run("exp", params={...}) as run: ...`."""
    return get_tracker().start_run(name, **kwargs)
