"""Единый интерфейс LLM проекта: ``complete(prompt, system=None, json_schema=None) -> str | dict``.

Провайдер — любой OpenAI-совместимый endpoint (OpenRouter, Ollama, vLLM ...):

* ``LLM_BASE_URL`` — по умолчанию ``https://openrouter.ai/api/v1``
* ``LLM_API_KEY``  — ключ (для Ollama можно любой непустой)
* ``LLM_MODEL``    — по умолчанию ``qwen/qwen3-8b``

:class:`MockLLM` — детерминированная заглушка для тестов/оффлайна, в сеть не ходит.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3-8b"


@runtime_checkable
class LLM(Protocol):
    name: str

    def complete(
        self, prompt: str, system: str | None = None, json_schema: dict | None = None
    ) -> str | dict: ...


class OpenAILLM:
    """Клиент OpenAI-совместимого API."""

    name = "openai"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 400,
    ):
        from openai import OpenAI  # ленивый импорт: тестам он не нужен

        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(
            base_url=base_url or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL,
            api_key=api_key or os.environ.get("LLM_API_KEY") or "no-key",
        )

    def complete(
        self, prompt: str, system: str | None = None, json_schema: dict | None = None
    ) -> str | dict:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {}
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema},
            }
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        text = resp.choices[0].message.content or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # qwen3 reasoning
        if json_schema is not None:
            return json.loads(text)
        return text


_FIELD_RE = re.compile(r"^\s*([A-Za-z_ ]+):\s*(.+)$", re.M)


class MockLLM:
    """Детерминированная заглушка: собирает ответ из полей ``Key: value`` в промпте.

    Все вызовы сохраняются в ``self.calls`` — удобно проверять промпт в тестах.
    Можно передать свою функцию ``responder(prompt, system) -> str``.
    """

    name = "mock"

    def __init__(self, responder: Callable[[str, str | None], str] | None = None):
        self.responder = responder
        self.calls: list[dict[str, Any]] = []

    def complete(
        self, prompt: str, system: str | None = None, json_schema: dict | None = None
    ) -> str | dict:
        self.calls.append({"prompt": prompt, "system": system, "json_schema": json_schema})
        if self.responder is not None:
            return self.responder(prompt, system)
        fields = {k.strip().lower(): v.strip() for k, v in _FIELD_RE.findall(prompt)}
        if json_schema is not None:
            return {"explanation": self._text(fields)}
        return self._text(fields)

    @staticmethod
    def _text(f: dict[str, str]) -> str:
        played = f.get("played move", "?")
        best = f.get("best move", "?")
        swing = f.get("eval swing", "?")
        motif = f.get("motif", "none")
        if f.get("language", "en").lower().startswith("ru"):
            return (
                f"[mock] Ход {played} ухудшил позицию (изменение оценки: {swing}). "
                f"Сильнее было {best}. Мотив: {motif}."
            )
        return (
            f"[mock] The move {played} worsened the position (eval swing: {swing}). "
            f"{best} was stronger. Motif: {motif}."
        )


def get_llm(kind: str = "auto") -> LLM | None:
    """``openai`` | ``mock`` | ``none`` | ``auto`` (openai при ``LLM_API_KEY``, иначе None)."""
    kind = kind.lower()
    if kind == "auto":
        kind = "openai" if os.environ.get("LLM_API_KEY") else "none"
    if kind == "none":
        return None
    if kind == "mock":
        return MockLLM()
    if kind == "openai":
        return OpenAILLM()
    raise ValueError(f"Неизвестный LLM: {kind!r} (ожидается openai|mock|none|auto)")


_default: LLM | None = None


def complete(prompt: str, system: str | None = None, json_schema: dict | None = None) -> str | dict:
    """Вызов LLM по умолчанию (OpenAI-совместимый, если есть ключ, иначе MockLLM)."""
    global _default
    if _default is None:
        _default = get_llm("auto") or MockLLM()
    return _default.complete(prompt, system=system, json_schema=json_schema)
