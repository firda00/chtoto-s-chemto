import json
import shlex

import httpx
import pytest
import respx
import yaml

from local_llm.backends.ollama import (
    OllamaClient,
    OllamaOptions,
    custom_model_name,
    options_from_preset,
    render_modelfile,
)
from local_llm.backends.vllm import (
    VLLMClient,
    config_from_preset,
    render_command,
    render_compose,
    render_shell,
)
from local_llm.mock_server import make_mock_transport
from local_llm.presets import get_preset

# --------------------------------------------------------------------------- Modelfile


def test_modelfile_from_preset_defaults():
    p = get_preset("qwen2.5-7b")
    text = render_modelfile(p, "q4_K_M")
    lines = text.splitlines()
    assert "FROM qwen2.5:7b-instruct-q4_K_M" in lines
    assert "PARAMETER num_ctx 8192" in lines
    assert "PARAMETER temperature 0.7" in lines
    assert "PARAMETER top_p 0.8" in lines
    assert "PARAMETER top_k 20" in lines
    assert "PARAMETER repeat_penalty 1.05" in lines
    assert not any("num_gpu" in ln for ln in lines)  # не задан — не пишем
    assert text.endswith("\n")
    assert "Оценка VRAM" in text


def test_modelfile_with_overrides_and_system():
    p = get_preset("llama3.1-8b")
    opts = options_from_preset(p, num_ctx=16384, num_gpu=99, temperature=0.2)
    text = render_modelfile(p, "q8_0", opts, system='Отвечай кратко. """')
    assert "FROM llama3.1:8b-instruct-q8_0" in text
    assert "PARAMETER num_ctx 16384" in text
    assert "PARAMETER num_gpu 99" in text
    assert "PARAMETER temperature 0.2" in text
    assert 'SYSTEM """Отвечай кратко. \\"\\"\\""""' in text


def test_modelfile_rejects_vllm_quant():
    with pytest.raises(ValueError, match="awq"):
        render_modelfile(get_preset("qwen2.5-7b"), "awq")


def test_ollama_options_validation_and_name():
    assert OllamaOptions(num_ctx=4096, seed=1).as_options() == {"num_ctx": 4096, "seed": 1}
    with pytest.raises(ValueError):
        OllamaOptions(temperature=5)
    assert custom_model_name(get_preset("qwen3-8b"), "q4_K_M", 8192) == "qwen3-8b-q4_k_m-8k"


# --------------------------------------------------------------------------- Ollama API


@respx.mock
def test_ollama_native_api_with_respx():
    tags = respx.get("http://ollama.test:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "qwen3:8b-q4_K_M"}]})
    )
    pull = respx.post("http://ollama.test:11434/api/pull").mock(
        return_value=httpx.Response(200, json={"status": "success"})
    )
    show = respx.post("http://ollama.test:11434/api/show").mock(
        return_value=httpx.Response(200, json={"details": {"quantization_level": "Q4_K_M"}})
    )
    with OllamaClient("ollama.test:11434") as cli:
        assert cli.host == "http://ollama.test:11434"
        assert cli.has_model("qwen3:8b-q4_K_M")
        assert not cli.has_model("llama3.2:3b")
        assert cli.pull("qwen3:8b-q4_K_M") == {"status": "success"}
        assert cli.show("qwen3:8b-q4_K_M")["details"]["quantization_level"] == "Q4_K_M"
    assert tags.call_count == 2
    assert json.loads(pull.calls.last.request.content) == {
        "model": "qwen3:8b-q4_K_M",
        "stream": False,
    }
    assert json.loads(show.calls.last.request.content) == {"model": "qwen3:8b-q4_K_M"}


@respx.mock
def test_ollama_chat_goes_through_openai_v1_with_options():
    route = respx.post("http://localhost:11434/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Ташкент"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )
    cli = OllamaClient("http://localhost:11434")
    opts = OllamaOptions(num_ctx=4096, temperature=0.1, num_gpu=99, num_predict=16)
    answer = cli.chat("qwen2.5:7b-instruct-q4_K_M", "Столица Узбекистана?", options=opts)
    assert answer == "Ташкент"
    body = json.loads(route.calls.last.request.content)
    assert body["temperature"] == 0.1
    assert body["max_tokens"] == 16
    assert body["options"] == {"num_ctx": 4096, "num_gpu": 99}


def test_ollama_errors_propagate():
    with OllamaClient("http://x", transport=make_mock_transport()) as cli:
        with pytest.raises(httpx.HTTPStatusError):
            cli.show("missing:model")
        assert next(m["name"] for m in cli.list_models()).startswith("qwen2.5")


# --------------------------------------------------------------------------- vLLM


def test_vllm_awq_command():
    cfg = config_from_preset(get_preset("qwen2.5-7b"), "awq", max_model_len=16384)
    cmd = render_command(cfg)
    assert cmd[:3] == ["vllm", "serve", "Qwen/Qwen2.5-7B-Instruct-AWQ"]
    joined = " ".join(cmd)
    assert "--quantization awq" in joined
    assert "--max-model-len 16384" in joined
    assert "--gpu-memory-utilization 0.9" in joined
    assert "--tensor-parallel-size 1" in joined
    assert "--served-model-name qwen2.5-7b" in joined
    assert "--dtype auto" in joined
    assert "--enable-prefix-caching" in cmd


def test_vllm_fp8_dynamic_vs_prequantized():
    dyn = config_from_preset(get_preset("qwen2.5-14b"), "fp8")
    assert dyn.model == "Qwen/Qwen2.5-14B-Instruct" and dyn.quantization == "fp8"
    pre = config_from_preset(get_preset("qwen3-8b"), "fp8")
    assert pre.model == "Qwen/Qwen3-8B-FP8" and pre.quantization is None
    fp16 = config_from_preset(get_preset("qwen3-8b"), "fp16")
    assert "--quantization" not in render_command(fp16)


def test_vllm_tp_kv_cache_and_options():
    cfg = config_from_preset(
        get_preset("llama3.1-8b"),
        "gptq",
        max_model_len=32768,
        gpu_memory_utilization=0.85,
        tensor_parallel_size=2,
        kv_cache_dtype="fp8",
        max_num_seqs=16,
        port=9000,
    )
    joined = " ".join(render_command(cfg))
    assert "--tensor-parallel-size 2" in joined
    assert "--kv-cache-dtype fp8" in joined
    assert "--max-num-seqs 16" in joined
    assert "--gpu-memory-utilization 0.85" in joined
    assert "--port 9000" in joined


def test_vllm_shell_roundtrip():
    cfg = config_from_preset(get_preset("qwen2.5-7b"), "awq")
    one = render_shell(cfg, multiline=False)
    assert shlex.split(one) == render_command(cfg)
    multi = render_shell(cfg)
    assert shlex.split(multi.replace("\\\n", " ")) == render_command(cfg)


def test_vllm_invalid_requests():
    with pytest.raises(ValueError, match="q4_K_M"):
        config_from_preset(get_preset("qwen2.5-7b"), "q4_K_M")
    with pytest.raises(ValueError, match="max_model_len"):
        config_from_preset(get_preset("qwen2.5-7b"), "awq", max_model_len=10**6)


def test_vllm_compose_is_valid_yaml():
    cfg = config_from_preset(get_preset("qwen2.5-7b"), "awq", tensor_parallel_size=2)
    doc = yaml.safe_load(render_compose(cfg))
    svc = doc["services"]["vllm"]
    assert svc["image"].startswith("vllm/vllm-openai")
    assert svc["command"] == cfg.args()
    assert svc["ports"] == ["8000:8000"]
    assert svc["ipc"] == "host"
    assert svc["deploy"]["resources"]["reservations"]["devices"][0]["count"] == 2
    assert "hf-cache" in doc["volumes"]


def test_vllm_client_against_mock():
    cli = VLLMClient("http://vllm.test:8000/v1", transport=make_mock_transport())
    assert cli.base_url == "http://vllm.test:8000"
    assert cli.health()
    assert "qwen2.5:7b-instruct-q4_K_M" in cli.models()
    assert cli.chat("qwen2.5-7b", "hi").startswith("[qwen2.5-7b]")
    cli.close()


def test_project_docker_compose_parses():
    from pathlib import Path

    doc = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "docker-compose.yml").read_text(encoding="utf-8")
    )
    assert {"ollama", "vllm"} <= set(doc["services"])
    assert doc["services"]["ollama"]["ports"] == ["11434:11434"]
    assert doc["services"]["vllm"]["profiles"] == ["vllm"]
