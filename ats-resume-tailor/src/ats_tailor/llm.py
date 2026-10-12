"""Единый интерфейс к LLM: complete(prompt, system=None, json_schema=None) -> str | dict.

* OpenAILLM — любой OpenAI-совместимый endpoint (OpenRouter, Ollama, vLLM).
* MockLLM — детерминированная заглушка без сети: разбирает JSON-блок INPUT из промпта
  и выполняет простые правила (переупорядочить навыки, поднять релевантные слова,
  собрать письмо по шаблону). Используется в тестах и в демо-режиме.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from typing import Any, Protocol

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3-8b"

INPUT_START = "<<<INPUT_JSON"
INPUT_END = "INPUT_JSON>>>"


class LLM(Protocol):
    name: str

    def complete(
        self, prompt: str, system: str | None = None, json_schema: dict | None = None
    ) -> str | dict: ...


def embed_input(payload: dict[str, Any]) -> str:
    """Встроить входные данные в промпт в машиночитаемом виде."""
    return f"{INPUT_START}\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n{INPUT_END}"


def extract_input(prompt: str) -> dict[str, Any] | None:
    pattern = re.escape(INPUT_START) + r"\s*(\{.*\})\s*" + re.escape(INPUT_END)
    m = re.search(pattern, prompt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def parse_json_response(text: str) -> dict:
    """Достать JSON-объект из ответа модели (с учётом ```json блоков и <think>)."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.S)
    if fence:
        cleaned = fence.group(1)
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


class OpenAILLM:
    """Клиент OpenAI-совместимого API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        timeout: float = 120.0,
    ) -> None:
        from openai import OpenAI

        self.base_url = base_url or os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
        self.temperature = temperature
        key = api_key or os.getenv("LLM_API_KEY") or "not-needed"
        self._client = OpenAI(base_url=self.base_url, api_key=key, timeout=timeout)
        self.name = f"openai-compatible:{self.model}"

    def complete(
        self, prompt: str, system: str | None = None, json_schema: dict | None = None
    ) -> str | dict:
        messages: list[dict[str, str]] = []
        sys_text = system or ""
        if json_schema is not None:
            sys_text += (
                "\n\nОтвечай ТОЛЬКО валидным JSON-объектом, соответствующим JSON Schema:\n"
                + json.dumps(json_schema, ensure_ascii=False)
            )
        if sys_text:
            messages.append({"role": "system", "content": sys_text.strip()})
        messages.append({"role": "user", "content": prompt})
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        if json_schema is not None:
            return parse_json_response(content)
        return content.strip()


Handler = Callable[[dict[str, Any]], str | dict]


class MockLLM:
    """Детерминированная заглушка. Никогда не ходит в сеть.

    * ``responses`` — очередь готовых ответов (для тестов конкретных сценариев);
    * иначе ответ строится встроенным обработчиком по полю ``task`` из INPUT-блока.
    """

    name = "mock"

    def __init__(
        self,
        responses: list[str | dict] | None = None,
        handlers: dict[str, Handler] | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []
        self._handlers: dict[str, Handler] = {
            "tailor_resume": _mock_tailor,
            "cover_letter": _mock_cover_letter,
            "extract_vacancy": lambda payload: {},
        }
        if handlers:
            self._handlers.update(handlers)

    def complete(
        self, prompt: str, system: str | None = None, json_schema: dict | None = None
    ) -> str | dict:
        self.calls.append({"prompt": prompt, "system": system, "json_schema": json_schema})
        if self._responses:
            return self._responses.pop(0)
        payload = extract_input(prompt) or {}
        handler = self._handlers.get(str(payload.get("task", "")))
        if handler is not None:
            return handler(payload)
        return {} if json_schema is not None else "MOCK RESPONSE"


# ---------------------------------------------------------------- mock handlers

_SPLIT_ITEMS = re.compile(r"\s*(?:,|;|\n|•|·|\|)\s*")


def split_skill_items(text: str) -> list[str]:
    items = []
    for raw in _SPLIT_ITEMS.split(text):
        item = raw.strip().lstrip("-*– ").strip()
        if item:
            items.append(item)
    return items


def _mock_tailor(payload: dict[str, Any]) -> dict:
    sections: dict[str, str] = dict(payload.get("sections", {}))
    present: list[str] = list(payload.get("present_keywords", []))
    in_skills: list[str] = list(payload.get("keywords_in_skills", []))
    lang = payload.get("language", "ru")
    title = payload.get("vacancy", {}).get("title", "")
    changes: list[dict[str, str]] = []

    skills = sections.get("skills", "")
    if skills:
        items = split_skill_items(skills)
        prio = [i for i in items if any(k.lower() in i.lower() for k in present)]
        rest = [i for i in items if i not in prio]
        to_add = [k for k in present if k not in in_skills]
        new_items = to_add + prio + rest
        if new_items != items:
            sections["skills"] = ", ".join(new_items)
            changes.append(
                {
                    "section": "skills",
                    "action": "reordered",
                    "detail": (
                        "Релевантные вакансии навыки перенесены в начало"
                        if lang == "ru"
                        else "Moved job-relevant skills to the top"
                    ),
                }
            )
        if to_add:
            prefix = (
                "Добавлены навыки, подтверждённые опытом: "
                if lang == "ru"
                else "Added skills evidenced in experience: "
            )
            changes.append(
                {
                    "section": "skills",
                    "action": "keywords_added",
                    "detail": prefix + ", ".join(to_add),
                }
            )

    summary = sections.get("summary", "")
    if summary and present and title:
        top = ", ".join(present[:5])
        tail = (
            f"Релевантно для позиции «{title}»: {top}."
            if lang == "ru"
            else f"Relevant for the {title} role: {top}."
        )
        if tail not in summary:
            sections["summary"] = summary.rstrip() + "\n\n" + tail
            changes.append(
                {
                    "section": "summary",
                    "action": "rephrased",
                    "detail": (
                        "Добавлен акцент на совпадающие с вакансией компетенции"
                        if lang == "ru"
                        else "Emphasised competencies matching the job"
                    ),
                }
            )
    return {"sections": sections, "changes": changes}


def _mock_cover_letter(payload: dict[str, Any]) -> str:
    lang = payload.get("language", "ru")
    name = payload.get("candidate_name") or ("Кандидат" if lang == "ru" else "Candidate")
    default_title = "вакансию" if lang == "ru" else "the role"
    title = payload.get("vacancy", {}).get("title") or default_title
    kws = ", ".join(payload.get("present_keywords", [])[:6])
    highlights = payload.get("highlights", [])[:3]
    if lang == "ru":
        lines = [
            "Здравствуйте!",
            "",
            f"Меня заинтересовала позиция «{title}». "
            f"Мой опыт напрямую пересекается с требованиями: {kws}.",
        ]
        if highlights:
            lines += ["", "Из того, что я уже делал:"] + [f"- {h}" for h in highlights]
        lines += [
            "",
            "Буду рад обсудить, чем могу быть полезен команде.",
            "",
            f"С уважением, {name}",
        ]
    else:
        lines = [
            "Dear Hiring Team,",
            "",
            f"I am excited to apply for the {title} position. "
            f"My background maps directly to your requirements: {kws}.",
        ]
        if highlights:
            lines += ["", "Highlights:"] + [f"- {h}" for h in highlights]
        lines += [
            "",
            "I would welcome the chance to discuss how I can help.",
            "",
            f"Best regards, {name}",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------- factory


def is_llm_configured() -> bool:
    """Настоящая LLM доступна: есть ключ или локальный endpoint, и мок не форсирован."""
    if os.getenv("LLM_MOCK", "").lower() in {"1", "true", "yes"}:
        return False
    base = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
    local = any(h in base for h in ("localhost", "127.0.0.1", "0.0.0.0"))
    return bool(os.getenv("LLM_API_KEY")) or local


def get_llm(mock: bool | None = None) -> LLM:
    """mock=True — всегда MockLLM; None — авто по переменным окружения."""
    if mock is True or (mock is None and not is_llm_configured()):
        return MockLLM()
    return OpenAILLM()
