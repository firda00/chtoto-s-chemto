import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from local_llm.cli import app

runner = CliRunner()
PROMPTS = str(Path(__file__).resolve().parents[1] / "data" / "prompts.txt")


def run(*args: str):
    result = runner.invoke(app, list(args))
    return result


def test_help_lists_commands():
    r = run("--help")
    assert r.exit_code == 0
    for cmd in ("presets", "estimate", "recommend", "modelfile", "vllm-cmd", "bench", "compare"):
        assert cmd in r.output


def test_presets_table_and_json():
    r = run("presets")
    assert r.exit_code == 0, r.output
    assert "qwen2.5-14b" in r.output and "q4_K_M" in r.output
    r = run("presets", "--backend", "vllm")
    assert r.exit_code == 0 and "awq" in r.output
    data = json.loads(run("presets", "--json").output)
    assert data["llama3.2-3b"]["params_b"] == 3.21


def test_estimate():
    r = run("estimate", "--model", "qwen2.5-7b", "--quant", "q4_K_M", "--ctx", "8192")
    assert r.exit_code == 0, r.output
    assert "Итого" in r.output and "5.66" in r.output
    data = json.loads(
        run("estimate", "--params", "8.03", "--quant", "q4_K_M", "--ctx", "8192", "--json").output
    )
    assert data["kv_cache_gb"] == pytest.approx(1.0, abs=1e-3)
    assert data["fits_90pct"] == {"8": True, "12": True, "16": True, "24": True}
    assert run("estimate", "--quant", "q4_K_M").exit_code != 0
    assert run("estimate", "--model", "qwen2.5-7b", "--quant", "q9").exit_code != 0


def test_recommend():
    r = run("recommend", "--vram", "12")
    assert r.exit_code == 0, r.output
    assert "Рекомендация" in r.output and "ollama pull" in r.output
    data = json.loads(run("recommend", "--vram", "24", "--target", "quality", "--json").output)
    assert data["best"]["preset_id"] == "qwen2.5-14b"
    r = run("recommend", "--vram", "16", "--backend", "vllm")
    assert r.exit_code == 0 and "vllm-cmd" in r.output
    assert run("recommend", "--vram", "2").exit_code == 1
    assert run("recommend", "--vram", "8", "--target", "fast").exit_code != 0


def test_modelfile(tmp_path):
    r = run("modelfile", "--preset", "qwen2.5-7b", "--quant", "q4_K_M", "--ctx", "8192")
    assert r.exit_code == 0, r.output
    assert "FROM qwen2.5:7b-instruct-q4_K_M" in r.output
    out = tmp_path / "Modelfile"
    r = run("modelfile", "--preset", "qwen3-8b", "--num-gpu", "99", "-o", str(out))
    assert r.exit_code == 0
    assert "PARAMETER num_gpu 99" in out.read_text(encoding="utf-8")
    assert "ollama create qwen3-8b-q4_k_m-8k" in r.output
    assert run("modelfile", "--preset", "qwen3-8b", "--quant", "q5_K_M").exit_code != 0
    assert run("modelfile", "--preset", "nope").exit_code != 0


def test_vllm_cmd():
    r = run("vllm-cmd", "--preset", "qwen2.5-7b", "--quant", "awq", "--max-model-len", "16384")
    assert r.exit_code == 0, r.output
    assert "vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ" in r.output
    assert "--quantization awq" in r.output
    r = run("vllm-cmd", "--preset", "qwen3-8b", "--quant", "fp8", "--compose", "--tp", "2")
    assert r.exit_code == 0
    doc = yaml.safe_load(r.stdout)
    assert doc["services"]["vllm"]["command"][0] == "Qwen/Qwen3-8B-FP8"
    assert run("vllm-cmd", "--preset", "qwen2.5-7b", "--quant", "q4_K_M").exit_code != 0


def test_bench_mock_single_and_compare(tmp_path):
    jp = tmp_path / "run.json"
    cp = tmp_path / "run.csv"
    r = run(
        "bench", "--mock", "--model", "qwen2.5:7b-instruct-q4_K_M", "--prompts", PROMPTS,
        "--limit", "3", "--repeats", "2", "--max-tokens", "16",
        "--json", str(jp), "--csv", str(cp),
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "decode tok/s" in r.output
    saved = json.loads(jp.read_text(encoding="utf-8"))
    assert saved["summary"]["n_requests"] == 6
    assert saved["summary"]["ttft_ms"]["p50"] == pytest.approx(180.0)
    assert saved["summary"]["decode_tps"]["mean"] == pytest.approx(45.0)
    assert cp.read_text(encoding="utf-8").count("\n") == 7

    r = run(
        "bench", "--mock", "--model", "qwen2.5:7b-instruct-q4_K_M",
        "--model-b", "qwen2.5:7b-instruct-q8_0", "--prompts", PROMPTS, "--limit", "2", "-q",
        "--json", str(tmp_path / "ab.json"),
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "Сравнение" in r.output
    r = run("compare", str(tmp_path / "ab_a.json"), str(tmp_path / "ab_b.json"))
    assert r.exit_code == 0 and "Δ B к A" in r.output


def test_bench_requires_model(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert run("bench", "--mock", "--prompts", PROMPTS).exit_code != 0


def test_bench_unreachable_server_exits_nonzero(monkeypatch):
    """Без --mock: подменяем транспорт на «connection refused», сеть не трогаем."""
    import httpx

    from local_llm import bench as bench_mod

    def refuse(request):
        raise httpx.ConnectError("connection refused")

    real_client = httpx.Client
    monkeypatch.setattr(
        bench_mod.httpx,
        "Client",
        lambda **kw: real_client(transport=httpx.MockTransport(refuse), timeout=kw.get("timeout")),
    )
    r = run(
        "bench", "--model", "m", "--base-url", "http://ollama.invalid/v1", "--prompts", PROMPTS,
        "--limit", "1", "--warmup", "0",
    )  # fmt: skip
    assert r.exit_code == 1


def test_ask_mock():
    r = run("ask", "Привет", "--mock", "--model", "demo")
    assert r.exit_code == 0 and r.output.startswith("[demo]")


def test_ollama_commands_against_mock(monkeypatch):
    import httpx

    from local_llm.backends import ollama as ollama_mod
    from local_llm.mock_server import make_mock_transport

    real_client = httpx.Client
    monkeypatch.setattr(
        ollama_mod.httpx,
        "Client",
        lambda **kw: real_client(transport=make_mock_transport(), base_url=kw["base_url"]),
    )
    r = run("ollama", "list")
    assert r.exit_code == 0 and "qwen2.5:7b-instruct-q4_K_M" in r.output
    r = run("ollama", "show", "qwen2.5:7b-instruct-q4_K_M")
    assert r.exit_code == 0 and "Q4_K_M" in r.output
    assert run("ollama", "show", "missing").exit_code == 2
