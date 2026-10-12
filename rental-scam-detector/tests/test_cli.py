from __future__ import annotations

import json

from typer.testing import CliRunner

from scam_detector.cli import app

from .conftest import SAMPLES

runner = CliRunner()


def test_cli_end_to_end(tmp_path):
    data = tmp_path / "ds.csv"
    r = runner.invoke(
        app, ["make-dataset", "--n", "300", "--scam-ratio", "0.3", "--out", str(data)]
    )
    assert r.exit_code == 0, r.output
    assert data.exists()

    r = runner.invoke(
        app, ["train", "--data", str(data), "--out-dir", str(tmp_path), "--iterations", "50"]
    )
    assert r.exit_code == 0, r.output
    model = tmp_path / "scam.cbm"
    assert model.exists()

    r = runner.invoke(
        app, ["check", str(SAMPLES / "listing_02.json"), "--model", str(model), "--json"]
    )
    assert r.exit_code == 0, r.output
    report = json.loads(r.output)
    assert report["listing_id"] == "listing_02" and 0 <= report["score"] <= 100

    out = tmp_path / "reports.json"
    r = runner.invoke(app, ["batch", str(SAMPLES), "--model", str(model), "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert len(json.loads(out.read_text(encoding="utf-8"))) == 6


def test_train_without_dataset_fails(tmp_path):
    r = runner.invoke(app, ["train", "--data", str(tmp_path / "missing.csv")])
    assert r.exit_code == 1
