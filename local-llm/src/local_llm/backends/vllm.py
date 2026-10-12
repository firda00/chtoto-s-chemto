"""vLLM: генерация команды `vllm serve` и docker-compose из пресета + клиент OpenAI API."""

from __future__ import annotations

import shlex
from typing import Any, Literal

import httpx
import yaml
from pydantic import BaseModel, Field

from local_llm.client import LLMClient, LLMSettings
from local_llm.presets import ModelPreset

VLLMQuant = Literal["awq", "gptq", "fp8"]
DEFAULT_IMAGE = "vllm/vllm-openai:latest"


class VLLMConfig(BaseModel):
    model: str = Field(description="HF-репозиторий или локальный путь")
    served_model_name: str | None = None
    dtype: Literal["auto", "float16", "bfloat16", "half"] = "auto"
    quantization: VLLMQuant | None = None
    max_model_len: int = Field(default=8192, gt=0)
    gpu_memory_utilization: float = Field(default=0.90, gt=0, le=1)
    tensor_parallel_size: int = Field(default=1, ge=1)
    kv_cache_dtype: Literal["auto", "fp8"] = "auto"
    max_num_seqs: int | None = Field(default=None, ge=1)
    enable_prefix_caching: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    extra_args: list[str] = Field(default_factory=list)

    def args(self) -> list[str]:
        out = [self.model]
        if self.served_model_name:
            out += ["--served-model-name", self.served_model_name]
        out += ["--host", self.host, "--port", str(self.port)]
        out += ["--dtype", self.dtype]
        if self.quantization:
            out += ["--quantization", self.quantization]
        out += [
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            f"{self.gpu_memory_utilization:g}",
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
        ]
        if self.kv_cache_dtype != "auto":
            out += ["--kv-cache-dtype", self.kv_cache_dtype]
        if self.max_num_seqs:
            out += ["--max-num-seqs", str(self.max_num_seqs)]
        if self.enable_prefix_caching:
            out.append("--enable-prefix-caching")
        return out + list(self.extra_args)


def config_from_preset(
    preset: ModelPreset,
    quant: str = "awq",
    *,
    max_model_len: int | None = None,
    gpu_memory_utilization: float = 0.90,
    tensor_parallel_size: int = 1,
    kv_cache_dtype: Literal["auto", "fp8"] = "auto",
    max_num_seqs: int | None = None,
    port: int = 8000,
) -> VLLMConfig:
    if quant not in preset.vllm.quants:
        raise ValueError(
            f"{preset.id}: квантование {quant!r} недоступно для vLLM "
            f"(есть: {', '.join(preset.vllm.quants)})"
        )
    repo = preset.vllm.repo(quant)
    quantization: VLLMQuant | None = None
    if quant in ("awq", "gptq"):
        quantization = quant  # type: ignore[assignment]
    elif quant == "fp8" and repo == preset.vllm.hf_repo:
        quantization = "fp8"  # динамическое FP8-квантование bf16-чекпоинта при загрузке
    # Готовые FP8/AWQ/GPTQ-чекпоинты vLLM распознаёт по quantization_config, но флаг не мешает.
    ctx = max_model_len or min(8192, preset.max_ctx)
    if ctx > preset.max_ctx:
        raise ValueError(f"{preset.id}: max_model_len {ctx} > max_ctx {preset.max_ctx}")
    return VLLMConfig(
        model=repo,
        served_model_name=preset.id,
        quantization=quantization,
        max_model_len=ctx,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        kv_cache_dtype=kv_cache_dtype,
        max_num_seqs=max_num_seqs,
        port=port,
    )


def render_command(cfg: VLLMConfig) -> list[str]:
    return ["vllm", "serve", *cfg.args()]


def render_shell(cfg: VLLMConfig, *, multiline: bool = True) -> str:
    parts = render_command(cfg)
    if not multiline:
        return shlex.join(parts)
    head = shlex.join(parts[:3])
    rest = parts[3:]
    lines = [head]
    i = 0
    while i < len(rest):
        if i + 1 < len(rest) and not rest[i + 1].startswith("--"):
            lines.append(f"  {rest[i]} {shlex.quote(rest[i + 1])}")
            i += 2
        else:
            lines.append(f"  {rest[i]}")
            i += 1
    return " \\\n".join(lines)


def render_compose(cfg: VLLMConfig, *, image: str = DEFAULT_IMAGE, gpus: int | None = None) -> str:
    """docker-compose.yml с одним сервисом vllm (образ vllm/vllm-openai)."""
    n_gpus = gpus or cfg.tensor_parallel_size
    service: dict[str, Any] = {
        "image": image,
        "command": cfg.args(),
        "ports": [f"{cfg.port}:{cfg.port}"],
        "ipc": "host",
        "environment": ["HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}"],
        "volumes": ["hf-cache:/root/.cache/huggingface"],
        "deploy": {
            "resources": {
                "reservations": {
                    "devices": [{"driver": "nvidia", "count": n_gpus, "capabilities": ["gpu"]}]
                }
            }
        },
        "restart": "unless-stopped",
    }
    doc = {"services": {"vllm": service}, "volumes": {"hf-cache": {}}}
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


class VLLMClient:
    """Клиент OpenAI-совместимого сервера vLLM (`/v1/*`, `/health`)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        *,
        api_key: str = "vllm",
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.api_key = api_key
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)
        self._has_transport = transport is not None

    def close(self) -> None:
        self._http.close()

    def health(self) -> bool:
        try:
            return self._http.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def models(self) -> list[str]:
        resp = self._http.get("/v1/models", headers={"Authorization": f"Bearer {self.api_key}"})
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]

    def llm(self, model: str) -> LLMClient:
        return LLMClient(
            LLMSettings(base_url=f"{self.base_url}/v1", api_key=self.api_key, model=model),
            http_client=self._http if self._has_transport else None,
        )

    def chat(self, model: str, prompt: str, *, system: str | None = None, **params: Any) -> str:
        return str(self.llm(model).complete(prompt, system=system, **params))
