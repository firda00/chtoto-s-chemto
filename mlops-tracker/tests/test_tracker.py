from __future__ import annotations

import random
import subprocess
from pathlib import Path

import pytest

from mlops_tracker import tracker as tracker_mod
from mlops_tracker.store import RunNotFoundError, Store
from mlops_tracker.tracker import Tracker, git_info


def test_run_logs_params_metrics_artifacts_texts(tracker: Tracker, tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("lr: 0.001\n", encoding="utf-8")
    with tracker.start_run("baseline", params={"lr": 1e-3, "model": "qwen"}, tags=["qa"]) as run:
        run.log_metric("accuracy", 0.7, step=1)
        run.log_metric("accuracy", 0.9, step=2)
        run.log_metrics({"loss": 0.3, "f1": 0.8}, step=2)
        run.log_param("batch_size", 16)
        run.log_artifact(cfg)
        run.log_text("prompt", "Ты — помощник. Отвечай кратко.")
        run.set_tag("baseline")

    rec = tracker.store.get_run(run.id)
    assert rec.status == "finished"
    assert rec.end_time is not None and rec.duration_s is not None
    assert rec.params == {"lr": 1e-3, "model": "qwen", "batch_size": 16}
    assert [p.value for p in rec.metrics["accuracy"]] == [0.7, 0.9]
    assert [p.step for p in rec.metrics["accuracy"]] == [1, 2]
    assert rec.last_metrics()["accuracy"] == 0.9
    assert rec.last_metrics()["loss"] == 0.3
    assert set(rec.tags) == {"qa", "baseline"}
    assert Path(rec.artifacts["config.yaml"]).read_text(encoding="utf-8") == "lr: 0.001\n"
    assert rec.texts["prompt"].startswith("Ты — помощник")


def test_reproducibility_snapshot_and_seed(tracker: Tracker) -> None:
    with tracker.start_run("seeded", seed=123) as run:
        first = random.random()
    with tracker.start_run("seeded-again", seed=123):
        second = random.random()
    assert first == second  # глобальный сид выставлен трекером
    rec = tracker.store.get_run(run.id)
    assert rec.seed == 123
    assert rec.env["python"].count(".") == 2
    assert "pydantic" in rec.env["packages"]
    assert "platform" in rec.env


def test_random_seed_generated_when_missing(tracker: Tracker) -> None:
    with tracker.start_run("auto-seed") as run:
        pass
    assert isinstance(tracker.store.get_run(run.id).seed, int)


def test_failed_run_records_error(tracker: Tracker) -> None:
    with pytest.raises(ValueError), tracker.start_run("boom") as run:
        raise ValueError("сломалось")
    rec = tracker.store.get_run(run.id)
    assert rec.status == "failed"
    assert "сломалось" in (rec.error or "")


def test_git_info_tolerates_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLT_DISABLE_GIT")

    def boom(*a, **kw):
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr(tracker_mod.subprocess, "run", boom)
    assert git_info() == {}


def test_git_info_reads_commit(monkeypatch: pytest.MonkeyPatch, tracker: Tracker) -> None:
    monkeypatch.delenv("MLT_DISABLE_GIT")

    def fake_run(cmd, **kw):
        out = "abc123def\n" if "rev-parse" in cmd else " M file.py\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(tracker_mod.subprocess, "run", fake_run)
    with tracker.start_run("with-git") as run:
        pass
    rec = tracker.store.get_run(run.id)
    assert rec.git_commit == "abc123def"
    assert rec.env["git"] == {"commit": "abc123def", "dirty": True}


def test_list_runs_filters_and_prefix_lookup(tracker: Tracker) -> None:
    with tracker.start_run("a", experiment="exp1", tags=["x"]) as ra:
        pass
    with tracker.start_run("b", experiment="exp2") as rb:
        pass
    store = tracker.store
    assert [r.id for r in store.list_runs()] == [rb.id, ra.id]
    assert [r.name for r in store.list_runs(experiment="exp1")] == ["a"]
    assert [r.name for r in store.list_runs(tag="x")] == ["a"]
    assert store.get_run(ra.id[:6]).id == ra.id
    with pytest.raises(RunNotFoundError):
        store.get_run("zzzz")


def test_best_run_and_compare(tracker: Tracker) -> None:
    ids = {}
    for name, acc, lat in (("r1", 0.6, 120.0), ("r2", 0.9, 300.0), ("r3", 0.8, 80.0)):
        with tracker.start_run(name, params={"temperature": acc / 10}) as run:
            run.log_metric("accuracy", 0.1, step=0)
            run.log_metric("accuracy", acc, step=1)  # берётся последнее значение
            run.log_metric("latency", lat)
        ids[name] = run.id
    with pytest.raises(RuntimeError), tracker.start_run("crashed") as bad:
        bad.log_metric("accuracy", 1.0)
        raise RuntimeError
    store: Store = tracker.store

    best, value = store.best_run("accuracy")
    assert (best.name, value) == ("r2", 0.9)  # упавший ран не участвует
    fastest, lat = store.best_run("latency", mode="min")
    assert (fastest.name, lat) == ("r3", 80.0)
    assert store.best_run("missing") is None

    cmp = store.compare_runs(ids["r1"], ids["r2"])
    assert cmp["params"]["temperature"]["changed"] is True
    assert cmp["metrics"]["accuracy"]["delta"] == pytest.approx(0.3)


def test_default_tracker_uses_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db = tmp_path / "env.db"
    monkeypatch.setenv("MLT_DB", str(db))
    monkeypatch.setattr(tracker_mod, "_default", None)
    with tracker_mod.start_run("via-shortcut") as run:
        run.log_metric("m", 1.0)
    assert db.exists()
    assert Store(db).get_run(run.id).name == "via-shortcut"
