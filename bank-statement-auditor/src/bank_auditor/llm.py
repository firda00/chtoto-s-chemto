"""Единый интерфейс к LLM: complete(prompt, system=None, json_schema=None) -> str | dict.

Провайдер — любой OpenAI-совместимый endpoint (OpenRouter, Ollama, vLLM).
В тестах используется MockLLM: детерминированный, без сети и ключей.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3-8b"


class LLM(Protocol):
    def complete(
        self, prompt: str, system: str | None = None, json_schema: dict | None = None
    ) -> str | dict: ...


def _extract_json(text: str) -> dict:
    """Достать JSON-объект из ответа модели (в т.ч. обёрнутый в ```json ... ```)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


class OpenAICompatLLM:
    """Клиент OpenAI-совместимого API. Ключ/endpoint/модель — из переменных окружения."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        from openai import OpenAI  # импорт здесь: тестам SDK не нужен

        self.model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
        self.temperature = temperature
        self.client = OpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL),
            api_key=api_key or os.getenv("LLM_API_KEY", ""),
            timeout=timeout,
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
                "json_schema": {"name": "result", "schema": json_schema, "strict": False},
            }
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, **kwargs
        )
        content = resp.choices[0].message.content or ""
        return _extract_json(content) if json_schema is not None else content


class MockLLM:
    """Детерминированная заглушка.

    rules: подстрока (в верхнем регистре) -> ответ (dict для json_schema, иначе str).
    Если ни одно правило не сработало — возвращает default (для json_schema) или "".
    Все вызовы складываются в self.calls — удобно проверять в тестах.
    """

    def __init__(
        self,
        rules: dict[str, dict | str] | None = None,
        default: dict | None = None,
    ) -> None:
        self.rules = {k.upper(): v for k, v in (rules or {}).items()}
        self.default = (
            default
            if default is not None
            else {
                "category": "other",
                "confidence": 0.2,
                "reason": "mock: нет правила",
            }
        )
        self.calls: list[dict[str, Any]] = []

    def complete(
        self, prompt: str, system: str | None = None, json_schema: dict | None = None
    ) -> str | dict:
        self.calls.append({"prompt": prompt, "system": system, "json_schema": json_schema})
        upper = prompt.upper()
        for needle, answer in self.rules.items():
            if needle in upper:
                return answer
        return dict(self.default) if json_schema is not None else ""


def get_llm(use_real: bool = True) -> LLM:
    """Реальный клиент, если задан LLM_API_KEY, иначе MockLLM."""
    if use_real and os.getenv("LLM_API_KEY"):
        return OpenAICompatLLM()
    return MockLLM()


def complete(prompt: str, system: str | None = None, json_schema: dict | None = None) -> str | dict:
    """Функция-обёртка по соглашению монорепо."""
    return get_llm().complete(prompt, system=system, json_schema=json_schema)
