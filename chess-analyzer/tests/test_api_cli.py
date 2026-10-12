from fastapi.testclient import TestClient
from typer.testing import CliRunner

from chess_analyzer.api import app as api_app
from chess_analyzer.cli import app as cli_app

SCHOLAR = """[Event "t"]
[White "A"]
[Black "B"]
[Result "1-0"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0
"""


def test_api_health():
    client = TestClient(api_app)
    assert client.get("/health").json()["status"] == "ok"


def test_api_analyze_fake_engine_mock_llm():
    client = TestClient(api_app)
    resp = client.post(
        "/analyze", json={"pgn": SCHOLAR, "engine": "fake", "depth": 2, "llm": "mock", "lang": "en"}
    )
    assert resp.status_code == 200
    games = resp.json()["games"]
    assert len(games) == 1
    g = games[0]
    assert g["white"]["name"] == "A" and g["result"] == "1-0"
    blunders = [m for m in g["moves"] if m["classification"] == "blunder"]
    assert blunders and blunders[0]["san"] == "Nf6"
    assert blunders[0]["explanation"].startswith("[mock]")
    assert blunders[0]["motifs"][0]["kind"] == "allowed_mate"


def test_api_rejects_bad_pgn():
    client = TestClient(api_app)
    resp = client.post("/analyze", json={"pgn": "1. e4 e5 2. Ke3 *", "engine": "fake"})
    assert resp.status_code == 422


def test_cli_analyze_smoke(tmp_path, data_dir):
    out = tmp_path / "out"
    result = CliRunner().invoke(
        cli_app,
        [
            "analyze",
            str(data_dir / "missed_mate.pgn"),
            "--engine",
            "fake",
            "--depth",
            "14",
            "--lang",
            "ru",
            "--llm",
            "mock",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "Hasty Hero" in html and "<svg" in html
    assert (out / "report.md").exists() and (out / "report.json").exists()


def test_cli_summary(data_dir):
    result = CliRunner().invoke(cli_app, ["summary", str(data_dir / "sample.pgn")])
    assert result.exit_code == 0, result.output
    assert "Legal" in result.output and "Novice" in result.output


def test_cli_summary_with_engine(data_dir):
    result = CliRunner().invoke(
        cli_app, ["summary", str(data_dir / "sample.pgn"), "--engine", "fake"]
    )
    assert result.exit_code == 0, result.output
    assert "ACPL" in result.output
