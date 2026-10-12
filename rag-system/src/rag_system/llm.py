"""Единая точка доступа к LLM.

Интерфейс по соглашениям монорепо:
    complete(prompt, system=None, json_schema=None) -> str | dict
плюс chat(messages, tools=None) -> AssistantMessage — для OpenAI-style function calling.

* OpenAILLM — любой OpenAI-совместимый endpoint (OpenRouter, Ollama, vLLM, OpenAI):
  LLM_BASE_URL, LLM_API_KEY, LLM_MODEL.
* MockLLM — детерминированная заглушка без сети. Отвечает по сценарию (очередь ответов),
  а если сценарий пуст — простыми эвристиками: экстрактивный ответ из <context>,
  echo для переписывания запроса, вызов calculator/get_date по шаблонам вопроса.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import deque
from collections.abc import Iterable, Sequence
from typing import Any, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from rag_system.text import tokenize

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3-8b"


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AssistantMessage(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class LLM(Protocol):
    name: str

    def complete(
        self, prompt: str, system: str | None = None, json_schema: dict[str, Any] | None = None
    ) -> str | dict[str, Any]: ...

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AssistantMessage: ...


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Убирает рассуждения reasoning-моделей (Qwen3 и др.) из ответа."""
    return _THINK.sub("", text or "").strip()


def parse_json(text: str) -> dict[str, Any]:
    """Достаёт первый JSON-объект из ответа модели (терпимо к ```json и тексту вокруг)."""
    text = strip_think(text)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start : i + 1])
                        if isinstance(value, dict):
                            return value
                    except json.JSONDecodeError:
                        break
                    break
        start = text.find("{", start + 1)
    raise ValueError(f"В ответе модели нет JSON-объекта: {text[:200]!r}")


# --------------------------------------------------------------------------------------------
# OpenAI-compatible
# --------------------------------------------------------------------------------------------
class OpenAILLM:
    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        from openai import OpenAI

        self.model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
        self.base_url = base_url or os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
        key = api_key or os.getenv("LLM_API_KEY") or "not-needed"  # Ollama/vLLM ключ не проверяют
        self.temperature = temperature
        self.client = OpenAI(base_url=self.base_url, api_key=key, timeout=timeout)

    def complete(
        self, prompt: str, system: str | None = None, json_schema: dict[str, Any] | None = None
    ) -> str | dict[str, Any]:
        sys_parts = [system] if system else []
        if json_schema is not None:
            sys_parts.append(
                "Return ONLY a JSON object that matches this JSON schema:\n"
                + json.dumps(json_schema, ensure_ascii=False)
            )
        messages: list[dict[str, Any]] = []
        if sys_parts:
            messages.append({"role": "system", "content": "\n\n".join(sys_parts)})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {}
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature, **kwargs
            )
        except Exception as exc:
            if not kwargs:
                raise
            # не все OpenAI-совместимые провайдеры поддерживают response_format
            log.info("response_format не поддержан (%s), повтор без него", exc)
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature
            )
        text = strip_think(resp.choices[0].message.content or "")
        return parse_json(text) if json_schema is not None else text

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AssistantMessage:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature, **kwargs
        )
        msg = resp.choices[0].message
        calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        return AssistantMessage(content=strip_think(msg.content or "") or None, tool_calls=calls)


# --------------------------------------------------------------------------------------------
# Mock
# --------------------------------------------------------------------------------------------
_CTX_BLOCK = re.compile(r"^\[(\d+)\]([^\n]*)\n(.*?)(?=^\[\d+\]|\Z)", re.DOTALL | re.MULTILINE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9«\"])")
_NUM = r"(?:\d{1,3}(?:[  ]\d{3})+|\d+(?:[.,]\d+)?)"
MATH_EXPR = re.compile(rf"\(?{_NUM}\)?(?:\s*[-+*/×^%]\s*\(?{_NUM}\)?)+")
DATE_WORDS = re.compile(
    r"\b(today|date|weekday|сегодня|дата|число|день недели|какой сегодня)\b", re.IGNORECASE
)


def _tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def extractive_answer(question: str, context: str) -> str | None:
    """Выбирает предложение контекста с максимальным пересечением терминов вопроса.

    Это «LLM для бедных»: детерминированно, без сети — достаточно, чтобы
    проверить retrieval и форматирование ответа с цитатами.
    """
    q_terms = set(tokenize(question))
    if not q_terms:
        return None
    best: tuple[float, int, int, str] | None = None
    for m in _CTX_BLOCK.finditer(context):
        n, header, body = int(m.group(1)), m.group(2), m.group(3)
        head_terms = set(tokenize(header))
        head_bonus = 1.0 * len(q_terms & head_terms)
        order = 0
        for line in body.splitlines():
            line = line.strip().lstrip("-*• ").strip()
            if not line or line.startswith("#") or (line in header):
                continue
            for sent in _SENT_SPLIT.split(line):
                overlap = len(q_terms & set(tokenize(sent)))
                if overlap == 0 and head_bonus == 0:
                    continue
                score = overlap + head_bonus + (0.1 if re.search(r"\d", sent) else 0.0)
                key = (score, -n, -order, sent)
                if best is None or key[:3] > best[:3]:
                    best = key
                order += 1
    if best is None:
        return None
    return f"{best[3]} [{-best[1]}]"


class MockLLM:
    """Детерминированная заглушка LLM.

    responses — сценарий: строки, dict (для json_schema) или AssistantMessage / dict с
    ключом "tool_calls" (для chat). Каждый вызов complete/chat снимает один элемент.
    Когда сценарий пуст — работают эвристики по умолчанию. Все вызовы пишутся в .calls.
    """

    name = "mock"

    def __init__(self, responses: Iterable[Any] | None = None) -> None:
        self.script: deque[Any] = deque(responses or [])
        self.calls: list[dict[str, Any]] = []
        self._ids = 0

    # -- complete --------------------------------------------------------------------------
    def complete(
        self, prompt: str, system: str | None = None, json_schema: dict[str, Any] | None = None
    ) -> str | dict[str, Any]:
        self.calls.append(
            {"method": "complete", "prompt": prompt, "system": system, "json_schema": json_schema}
        )
        if self.script:
            item = self.script.popleft()
            if isinstance(item, AssistantMessage):
                item = item.content or ""
            if json_schema is not None and isinstance(item, str):
                return parse_json(item)
            return item
        if json_schema is not None:
            return {}  # «модель не справилась» → вызывающий код использует fallback
        query = _tag(prompt, "query")
        if query is not None:  # переписывание запроса: возвращаем как есть
            return query
        question = _tag(prompt, "question") or prompt
        context = _tag(prompt, "context") or ""
        return extractive_answer(question, context) or "В контексте нет ответа на этот вопрос."

    # -- chat ------------------------------------------------------------------------------
    def _next_id(self) -> str:
        self._ids += 1
        return f"call_{self._ids}"

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AssistantMessage:
        self.calls.append({"method": "chat", "messages": list(messages), "tools": tools})
        if self.script:
            item = self.script.popleft()
            if isinstance(item, AssistantMessage):
                return item
            if isinstance(item, dict):
                calls = [
                    ToolCall(
                        id=c.get("id") or self._next_id(),
                        name=c["name"],
                        arguments=c.get("arguments", {}),
                    )
                    for c in item.get("tool_calls", [])
                ]
                return AssistantMessage(content=item.get("content"), tool_calls=calls)
            return AssistantMessage(content=str(item))
        return self._default_chat(messages, tools)

    def _default_chat(
        self, messages: Sequence[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> AssistantMessage:
        user_idx = max(i for i, m in enumerate(messages) if m["role"] == "user")
        user = str(messages[user_idx]["content"])
        question = _tag(user, "question") or user
        context = _tag(user, "context") or ""
        tool_names = {t["function"]["name"] for t in tools or []}
        tool_msgs = [m for m in messages[user_idx + 1 :] if m["role"] == "tool"]

        if not tool_msgs:
            expr = MATH_EXPR.search(question)
            if "calculator" in tool_names and expr:
                return AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            id=self._next_id(),
                            name="calculator",
                            arguments={"expression": expr.group(0)},
                        )
                    ]
                )
            if "get_date" in tool_names and DATE_WORDS.search(question):
                return AssistantMessage(
                    tool_calls=[ToolCall(id=self._next_id(), name="get_date", arguments={})]
                )

        # финальный ответ: контекст + результаты search_docs; калькулятор/дата — в начало
        names = {}
        for m in messages[user_idx + 1 :]:
            for tc in m.get("tool_calls") or []:
                names[tc["id"]] = tc["function"]["name"]
        extra_ctx, facts = [], []
        for m in tool_msgs:
            name = names.get(m.get("tool_call_id"), "tool")
            if name == "search_docs":
                extra_ctx.append(str(m["content"]))
            else:
                facts.append(f"{name}: {m['content']}")
        full_ctx = "\n\n".join([context, *extra_ctx]).strip()
        answer = extractive_answer(question, full_ctx)
        parts = [*facts, answer] if answer else facts
        return AssistantMessage(
            content=" ".join(parts) if parts else "В контексте нет ответа на этот вопрос."
        )


# --------------------------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------------------------
def _is_local(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or host.endswith(".local")


def resolve_provider(provider: str | None = None) -> str:
    provider = (provider or os.getenv("LLM_PROVIDER") or "auto").lower()
    if provider != "auto":
        return provider
    base = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
    return "openai" if os.getenv("LLM_API_KEY") or _is_local(base) else "mock"


def get_llm(provider: str | None = None) -> LLM:
    """mock | openai | auto (openai при наличии LLM_API_KEY или локального endpoint)."""
    resolved = resolve_provider(provider)
    if resolved == "mock":
        return MockLLM()
    if resolved in {"openai", "openrouter", "ollama"}:
        return OpenAILLM()
    raise ValueError(f"Неизвестный LLM_PROVIDER: {resolved!r} (mock | openai | auto)")
