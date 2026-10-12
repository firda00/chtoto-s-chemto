import json

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from rag_system.api import create_app
from rag_system.cli import app

from .conftest import DOCS_DIR, EVAL_PATH


def test_cli_ingest_ask_eval(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # чтобы не подхватить чужой .env
    index = str(tmp_path / "idx")
    runner = CliRunner()

    r = runner.invoke(app, ["--mock", "--index-dir", index, "ingest", str(DOCS_DIR)])
    assert r.exit_code == 0, r.output
    assert "7 документов" in r.output

    r = runner.invoke(
        app, ["--mock", "--index-dir", index, "ask", "Сколько стоит гостевой день?", "-m", "naive"]
    )
    assert r.exit_code == 0, r.output
    assert "120 000" in r.output
    assert "pricing.md" in r.output

    r = runner.invoke(app, ["--index-dir", index, "ask", "Сколько будет 2 + 3?", "--mock",
                            "--json"])
    assert r.exit_code == 0, r.output
    payload = json.loads(r.output)
    assert payload["tool_calls"][0]["name"] == "calculator"
    assert payload["tool_calls"][0]["result"] == "5"

    r = runner.invoke(app, ["--mock", "--index-dir", index, "eval", "--data", str(EVAL_PATH)])
    assert r.exit_code == 0, r.output
    assert "naive" in r.output and "modular" in r.output


def test_cli_ask_on_empty_index(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(app, ["--mock", "--index-dir", str(tmp_path / "x"), "ask", "q"])
    assert r.exit_code == 1


def test_api(runtime) -> None:
    client = TestClient(create_app(lambda: runtime))
    health = client.get("/health").json()
    assert health["chunks"] > 0
    assert health["llm"] == "mock"

    r = client.post("/ask", json={"question": "What is the Wi-Fi speed?", "mode": "naive"})
    assert r.status_code == 200
    body = r.json()
    assert "500 Mbps" in body["answer"]
    assert body["citations"][0]["source"] == "faq.md"

    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ingest", json={"path": "no/such/dir"}).status_code == 404
