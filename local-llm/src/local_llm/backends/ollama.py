"""Ollama: нативный HTTP API (/api/tags, /api/pull, /api/show), генерация через
OpenAI-совместимый /v1 и рендер Modelfile из пресета."""

from __future__ import annotations

import os
from typing import Any

import httpx
from pydantic import BaseModel, Field

from local_llm.client import LLMClient, LLMSettings
from local_llm.presets import ModelPreset, get_quant

DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class OllamaOptions(BaseModel):
    """Параметры Modelfile (`PARAMETER ...`) и поля `options` нативного API."""

    num_ctx: int | None = Field(default=None, gt=0, description="Длина контекста (KV-кэш)")
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1)
    repeat_penalty: float | None = Field(default=None, ge=0.5, le=2)
    num_gpu: int | None = Field(default=None, ge=0, description="Слоёв на GPU (0=CPU, 99=все)")
    num_thread: int | None = Field(default=None, ge=1)
    num_predict: int | None = Field(default=None)
    seed: int | None = None

    def as_options(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


def options_from_preset(
    preset: ModelPreset,
    *,
    num_ctx: int | None = None,
    num_gpu: int | None = None,
    **overrides: Any,
) -> OllamaOptions:
    g = preset.generation
    data: dict[str, Any] = {
        "num_ctx": num_ctx or min(8192, preset.max_ctx),
        "temperature": g.temperature,
        "top_p": g.top_p,
        "top_k": g.top_k,
        "repeat_penalty": g.repeat_penalty,
        "num_gpu": num_gpu,
    }
    data.update({k: v for k, v in overrides.items() if v is not None})
    return OllamaOptions(**data)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def render_modelfile(
    preset: ModelPreset,
    quant: str = "q4_K_M",
    options: OllamaOptions | None = None,
    *,
    system: str | None = None,
) -> str:
    """Modelfile для `ollama create <name> -f Modelfile`."""
    if quant not in preset.ollama.quants:
        raise ValueError(
            f"{preset.id}: квантование {quant!r} недоступно в Ollama "
            f"(есть: {', '.join(preset.ollama.quants)})"
        )
    opts = options or options_from_preset(preset)
    est = preset.estimate(quant, opts.num_ctx or 2048)
    lines = [
        f"# {preset.display_name} · {quant} ({get_quant(quant).bits:g} бит/вес)",
        f"# Оценка VRAM: {est.total_gb:.1f} GiB "
        f"(веса {est.weights_gb:.1f} + KV {est.kv_cache_gb:.1f} + runtime {est.runtime_gb:.1f})",
        f"FROM {preset.ollama.tag(quant)}",
        "",
    ]
    for key, value in opts.as_options().items():
        lines.append(f"PARAMETER {key} {_format_value(value)}")
    if system:
        escaped = system.replace('"""', '\\"\\"\\"')
        lines += ["", f'SYSTEM """{escaped}"""']
    return "\n".join(lines) + "\n"


def custom_model_name(preset: ModelPreset, quant: str, num_ctx: int | None) -> str:
    suffix = f"-{num_ctx // 1024}k" if num_ctx else ""
    return f"{preset.id}-{quant.lower()}{suffix}"


class OllamaClient:
    """Клиент нативного API Ollama + OpenAI-совместимый /v1 для генерации."""

    def __init__(
        self,
        host: str | None = None,
        *,
        timeout: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self._http = httpx.Client(base_url=self.host, timeout=timeout, transport=transport)
        self._has_transport = transport is not None

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- нативный API ----------------------------------------------------
    def list_models(self) -> list[dict[str, Any]]:
        resp = self._http.get("/api/tags")
        resp.raise_for_status()
        return resp.json().get("models", [])

    def has_model(self, name: str) -> bool:
        return any(m.get("name") == name or m.get("model") == name for m in self.list_models())

    def pull(self, model: str) -> dict[str, Any]:
        resp = self._http.post("/api/pull", json={"model": model, "stream": False})
        resp.raise_for_status()
        return resp.json()

    def show(self, model: str) -> dict[str, Any]:
        resp = self._http.post("/api/show", json={"model": model})
        resp.raise_for_status()
        return resp.json()

    # --- OpenAI-совместимый /v1 -------------------------------------------
    @property
    def openai_base_url(self) -> str:
        return f"{self.host}/v1"

    def llm(self, model: str) -> LLMClient:
        return LLMClient(
            LLMSettings(base_url=self.openai_base_url, api_key="ollama", model=model),
            http_client=self._http if self._has_transport else None,
        )

    def chat(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        options: OllamaOptions | None = None,
    ) -> str:
        """Генерация через /v1/chat/completions; Ollama-специфичные опции — в `extra_body`."""
        params: dict[str, Any] = {}
        if options is not None:
            opts = options.as_options()
            for key in ("temperature", "top_p", "seed"):
                if key in opts:
                    params[key] = opts.pop(key)
            if "num_predict" in opts:
                params["max_tokens"] = opts.pop("num_predict")
            if opts:
                params["extra_body"] = {"options": opts}
        return str(self.llm(model).complete(prompt, system=system, **params))
