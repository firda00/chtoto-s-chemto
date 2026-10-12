"""Единый интерфейс к LLM (см. CONVENTIONS.md): `complete(prompt, system=None, json_schema=None)`.

* `OpenAICompatibleLLM` — любой OpenAI-совместимый endpoint (OpenRouter, Ollama, vLLM).
* `MockLLM` — детерминированная заглушка: тесты и демо никогда не ходят в сеть.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import threading
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3-8b"

Completion = str | dict[str, Any]
Handler = Callable[[str, str | None, dict[str, Any] | None], Completion]


@runtime_checkable
class LLM(Protocol):
    model: str
    last_usage: dict[str, int]

    def complete(
        self, prompt: str, system: str | None = None, json_schema: dict[str, Any] | None = None
    ) -> Completion: ...


def count_tokens(text: str) -> int:
    """Грубая оценка числа токенов (≈ слова + пунктуация) — достаточно для mock-учёта стоимости."""
    return len(re.findall(r"\w+|[^\w\s]", text))


def extract_json(text: str) -> Any:
    """Достаёт JSON из ответа модели: снимает ```json-ограждения, ищет первый {...} / [...]."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"В ответе нет валидного JSON: {text[:200]!r}")


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        from openai import OpenAI

        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
        self.temperature = temperature
        self.client = OpenAI(
            base_url=base_url or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL,
            api_key=api_key or os.environ.get("LLM_API_KEY") or "not-set",
            timeout=timeout,
        )
        self.last_usage: dict[str, int] = {}

    def complete(
        self, prompt: str, system: str | None = None, json_schema: dict[str, Any] | None = None
    ) -> Completion:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if json_schema is not None:
            prompt = (
                f"{prompt}\n\nОтветь ТОЛЬКО JSON-объектом по схеме:\n"
                f"{json.dumps(json_schema, ensure_ascii=False)}"
            )
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": json_schema},
            }
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception:
            if json_schema is None:
                raise
            # Не все провайдеры умеют json_schema — откатываемся на инструкцию в промпте.
            kwargs.pop("response_format")
            resp = self.client.chat.completions.create(**kwargs)
        usage = resp.usage
        self.last_usage = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }
        content = resp.choices[0].message.content or ""
        return extract_json(content) if json_schema is not None else content


def _stub_from_schema(schema: dict[str, Any]) -> Any:
    """Минимальный валидный объект по JSON Schema (для MockLLM без обработчика)."""
    if "enum" in schema:
        return schema["enum"][0]
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "null")
    if t == "object":
        props = schema.get("properties", {})
        return {k: _stub_from_schema(v) for k, v in props.items()}
    if t == "array":
        return []
    if t == "integer":
        return int(schema.get("minimum", 0))
    if t == "number":
        return float(schema.get("minimum", 0))
    if t == "boolean":
        return False
    if t == "null":
        return None
    return ""


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 1 or w.isdigit()}


def mock_judge(prompt: str) -> dict[str, Any]:
    """Эвристический «судья» для MockLLM: доля слов эталона, найденных в ответе → оценка 1..5.

    Понимает промпт, собранный `llm_judge` (теги <expected> и <output>)."""
    exp = re.search(r"<expected>(.*?)</expected>", prompt, re.S)
    out = re.search(r"<output>(.*?)</output>", prompt, re.S)
    expected = _words(exp.group(1)) if exp else set()
    output = _words(out.group(1)) if out else set()
    if not output:
        return {"score": 1, "reason": "mock: пустой ответ"}
    if not expected:
        return {"score": 3, "reason": "mock: нет эталона, нейтральная оценка"}
    recall = len(expected & output) / len(expected)
    score = 1 + round(recall * 4)
    return {"score": score, "reason": f"mock: совпало {recall:.0%} слов эталона"}


class MockLLM:
    """Детерминированная заглушка.

    Приоритет: `handler(prompt, system, json_schema)` → очередь `responses` → встроенная логика
    (оценка судьи по пересечению слов для промптов `llm_judge`, заглушка по схеме, эхо).
    Все вызовы пишутся в `calls` — удобно проверять промпты в тестах.
    """

    def __init__(
        self,
        responses: list[Completion] | None = None,
        handler: Handler | None = None,
        model: str = "mock-llm",
    ) -> None:
        self.model = model
        self.handler = handler
        self._responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []
        self.last_usage: dict[str, int] = {}
        self._lock = threading.Lock()

    def complete(
        self, prompt: str, system: str | None = None, json_schema: dict[str, Any] | None = None
    ) -> Completion:
        with self._lock:
            self.calls.append({"prompt": prompt, "system": system, "json_schema": json_schema})
            if self.handler is not None:
                result = self.handler(prompt, system, json_schema)
            elif self._responses:
                result = self._responses.pop(0)
            elif (
                json_schema is not None
                and "<output>" in prompt
                and "score" in json.dumps(json_schema)
            ):
                result = mock_judge(prompt)
            elif json_schema is not None:
                result = _stub_from_schema(json_schema)
            else:
                result = f"mock: {prompt.strip()[:80]}"
            if json_schema is not None and isinstance(result, str):
                # Не JSON — отдаём строку как есть: вызывающий код сам решает, как разбирать.
                with contextlib.suppress(ValueError):
                    result = extract_json(result)
            text_out = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
            self.last_usage = {
                "prompt_tokens": count_tokens((system or "") + prompt),
                "completion_tokens": count_tokens(text_out),
            }
            return result


def get_llm(provider: str | None = None, mock_handler: Handler | None = None) -> LLM:
    """Фабрика: `mock` | `openai` | `auto` (по умолчанию — реальная модель, только если задан
    `LLM_API_KEY`; иначе MockLLM)."""
    provider = (provider or os.environ.get("LLM_PROVIDER") or "auto").lower()
    if provider == "openai" or (provider == "auto" and os.environ.get("LLM_API_KEY")):
        return OpenAICompatibleLLM()
    if provider not in ("auto", "mock", "openai"):
        raise ValueError(f"Неизвестный LLM-провайдер: {provider}")
    return MockLLM(handler=mock_handler)
