"""Единый OpenAI-совместимый клиент (см. CONVENTIONS.md) и детерминированная заглушка MockLLM.

Переменные окружения:
    LLM_BASE_URL  — по умолчанию https://openrouter.ai/api/v1 (Ollama: http://localhost:11434/v1)
    LLM_API_KEY   — для Ollama/vLLM подойдёт любая строка
    LLM_MODEL     — по умолчанию qwen/qwen3-8b
    LLM_MOCK=1    — использовать MockLLM (без сети)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator
from typing import Any, Protocol

import httpx
from openai import OpenAI
from pydantic import BaseModel, Field

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3-8b"
OLLAMA_OPENAI_URL = "http://localhost:11434/v1"
VLLM_OPENAI_URL = "http://localhost:8000/v1"

JsonSchema = dict[str, Any] | type[BaseModel]


class LLMSettings(BaseModel):
    base_url: str = DEFAULT_BASE_URL
    api_key: str = Field(default="", repr=False)
    model: str = DEFAULT_MODEL
    timeout: float = 120.0

    @classmethod
    def from_env(cls, **overrides: Any) -> LLMSettings:
        data: dict[str, Any] = {
            "base_url": os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL,
            "api_key": os.environ.get("LLM_API_KEY") or "",
            "model": os.environ.get("LLM_MODEL") or DEFAULT_MODEL,
        }
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)


class SupportsComplete(Protocol):
    def complete(
        self, prompt: str, system: str | None = None, json_schema: JsonSchema | None = None
    ) -> str | dict[str, Any]: ...


def _schema_dict(json_schema: JsonSchema) -> dict[str, Any]:
    if isinstance(json_schema, type) and issubclass(json_schema, BaseModel):
        return json_schema.model_json_schema()
    return dict(json_schema)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json_answer(text: str) -> dict[str, Any]:
    """Достаёт JSON из ответа модели (локальные модели любят оборачивать его в ```json)."""
    cleaned = _FENCE.sub("", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": system}] if system else []
    msgs.append({"role": "user", "content": prompt})
    return msgs


class LLMClient:
    """Тонкая обёртка над `openai.OpenAI` для любого OpenAI-совместимого endpoint."""

    def __init__(
        self,
        settings: LLMSettings | None = None,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.settings = settings or LLMSettings.from_env()
        # Явно передаём httpx.Client: SDK openai>=3 по умолчанию использует httpx2, а нам нужен
        # один HTTP-стек (httpx) для всего проекта — и для мокирования через respx в тестах.
        self._client = OpenAI(
            base_url=self.settings.base_url,
            api_key=self.settings.api_key or "local",
            timeout=self.settings.timeout,
            http_client=http_client or httpx.Client(timeout=self.settings.timeout),
            max_retries=2,
        )

    @property
    def model(self) -> str:
        return self.settings.model

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_schema: JsonSchema | None = None,
        **params: Any,
    ) -> str | dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model, "messages": _messages(prompt, system)}
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": _schema_dict(json_schema)},
            }
        kwargs.update(params)
        resp = self._client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        return parse_json_answer(text) if json_schema is not None else text

    def stream(self, prompt: str, system: str | None = None, **params: Any) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model, messages=_messages(prompt, system), stream=True, **params
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def _fake_value(schema: dict[str, Any], defs: dict[str, Any]) -> Any:
    if "$ref" in schema:
        return _fake_value(defs.get(schema["$ref"].split("/")[-1], {}), defs)
    if "enum" in schema:
        return schema["enum"][0]
    if "default" in schema:
        return schema["default"]
    for key in ("anyOf", "oneOf", "allOf"):
        if schema.get(key):
            return _fake_value(schema[key][0], defs)
    match schema.get("type"):
        case "object":
            props = schema.get("properties", {})
            return {name: _fake_value(sub, defs) for name, sub in props.items()}
        case "array":
            return []
        case "string":
            return "mock"
        case "integer":
            return 0
        case "number":
            return 0.0
        case "boolean":
            return False
        case _:
            return None


class MockLLM:
    """Детерминированная заглушка: одинаковый вход → одинаковый ответ, без сети."""

    def __init__(self, model: str = "mock-llm", responses: dict[str, str] | None = None) -> None:
        self.model = model
        self.responses = responses or {}
        self.calls: list[dict[str, Any]] = []

    def answer_text(self, prompt: str, n_words: int = 24) -> str:
        if prompt in self.responses:
            return self.responses[prompt]
        digest = hashlib.sha256(f"{self.model}|{prompt}".encode()).hexdigest()
        words = [f"w{digest[i % 60 : i % 60 + 4]}" for i in range(n_words)]
        return f"[{self.model}] " + " ".join(words)

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_schema: JsonSchema | None = None,
        **params: Any,
    ) -> str | dict[str, Any]:
        self.calls.append({"prompt": prompt, "system": system, "json_schema": json_schema})
        if json_schema is not None:
            schema = _schema_dict(json_schema)
            return _fake_value(schema, schema.get("$defs", {}))
        return self.answer_text(prompt)

    def stream(self, prompt: str, system: str | None = None, **params: Any) -> Iterator[str]:
        text = str(self.complete(prompt, system))
        for i, word in enumerate(text.split(" ")):
            yield word if i == 0 else " " + word


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def get_llm(mock: bool | None = None, **overrides: Any) -> LLMClient | MockLLM:
    """Фабрика: MockLLM при `mock=True` или `LLM_MOCK=1`, иначе реальный клиент."""
    if mock is None:
        mock = _env_flag("LLM_MOCK")
    settings = LLMSettings.from_env(**overrides)
    if mock:
        return MockLLM(model=settings.model)
    return LLMClient(settings)


def complete(
    prompt: str, system: str | None = None, json_schema: JsonSchema | None = None
) -> str | dict[str, Any]:
    """Единый интерфейс проекта: `complete(prompt, system=None, json_schema=None) -> str | dict`."""
    return get_llm().complete(prompt, system=system, json_schema=json_schema)
