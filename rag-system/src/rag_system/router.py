"""Маршрутизация запроса: в какую коллекцию искать и какие инструменты предложить LLM.

* LLMRouter — просит LLM вернуть JSON {collection, tools, reason}; ответ валидируется
  Pydantic-схемой. Невалидный JSON / неизвестная коллекция → fallback на KeywordRouter.
* KeywordRouter — детерминированные правила по стемам ключевых слов (RU + EN).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from rag_system.llm import DATE_WORDS, LLM, MATH_EXPR
from rag_system.text import tokenize

log = logging.getLogger(__name__)

ALL = "all"
KNOWN_TOOLS = ("calculator", "get_date")

# Префиксы стемов: токен запроса совпадает, если начинается с одного из них.
DEFAULT_KEYWORDS: dict[str, list[str]] = {
    "pricing": [
        "цен",
        "сто",
        "стои",
        "тариф",
        "скидк",
        "абонем",
        "оплат",
        "price",
        "cost",
        "tariff",
        "discount",
        "fee",
        "flex",
        "fixed",
        "офис",
        "переговор",
        "day",
    ],
    "schedule": [
        "расписан",
        "час",
        "работ",
        "открыт",
        "праздн",
        "мероприят",
        "нетворк",
        "лекци",
        "schedul",
        "hour",
        "open",
        "event",
        "holiday",
        "суббот",
        "воскрес",
        "ресепш",
    ],
    "faq": [
        "wi",
        "fi",
        "печат",
        "принтер",
        "print",
        "park",
        "парков",
        "guest",
        "locker",
        "шкаф",
        "kitchen",
        "кухн",
        "speed",
        "скорост",
    ],
    "policies": [
        "правил",
        "polic",
        "refund",
        "возврат",
        "pet",
        "dog",
        "собак",
        "животн",
        "cancel",
        "отмен",
        "quiet",
        "тих",
        "security",
        "rule",
    ],
    "company": [
        "компан",
        "основ",
        "адрес",
        "команд",
        "поддерж",
        "контакт",
        "support",
        "contact",
        "found",
        "team",
        "площадк",
        "телеграм",
    ],
    "services": [
        "legal",
        "юридич",
        "mail",
        "почт",
        "посылк",
        "parcel",
        "coffee",
        "кофе",
        "catering",
        "service",
        "услуг",
    ],
}

DEFAULT_DESCRIPTIONS: dict[str, str] = {
    "pricing": "prices, tariffs, memberships, meeting room rates, discounts",
    "schedule": "opening hours, reception hours, holidays, regular events",
    "faq": "Wi-Fi, printing, parking, lockers, kitchen, guests",
    "policies": "refunds, cancellations, pets, quiet zone, security rules",
    "company": "about the company, address, team, support contacts and response time",
    "services": "legal address, mail handling, event hosting, coffee bar",
}


class Route(BaseModel):
    """Итоговое решение роутера. collection=None — искать по всем коллекциям."""

    collection: str | None = None
    tools: list[str] = Field(default_factory=list)
    reason: str = ""
    router: str = "keyword"


class RouteDecision(BaseModel):
    """Строгая схема ответа LLM (все поля обязательны — пустой {} не пройдёт)."""

    collection: str = Field(description="One of the listed collections, or 'all'")
    tools: list[str] = Field(description="Subset of: calculator, get_date. Empty if not needed")
    reason: str = Field(default="", description="Short justification")


ROUTE_SCHEMA: dict[str, Any] = RouteDecision.model_json_schema()


class Router(Protocol):
    def route(self, query: str, collections: Sequence[str]) -> Route: ...


def suggest_tools(query: str) -> list[str]:
    tools = []
    if MATH_EXPR.search(query):
        tools.append("calculator")
    if DATE_WORDS.search(query):
        tools.append("get_date")
    return tools


def _match(token: str, prefix: str) -> bool:
    # стемы обрезаются до 7 символов, поэтому длинный префикс может быть длиннее токена
    return token.startswith(prefix) or (len(token) >= 5 and prefix.startswith(token))


class KeywordRouter:
    def __init__(self, keywords: Mapping[str, Sequence[str]] | None = None) -> None:
        self.keywords = {
            k: [p.lower() for p in v] for k, v in (keywords or DEFAULT_KEYWORDS).items()
        }

    def scores(self, query: str, collections: Sequence[str]) -> dict[str, int]:
        toks = tokenize(query)
        out: dict[str, int] = {}
        for coll in collections:
            prefixes = self.keywords.get(coll, [coll.lower()])
            out[coll] = sum(1 for t in toks if any(_match(t, p) for p in prefixes))
        return out

    def route(self, query: str, collections: Sequence[str]) -> Route:
        scores = self.scores(query, collections)
        best = max(scores.values(), default=0)
        winners = [c for c, s in scores.items() if s == best]
        # неоднозначность (ничья) или ноль совпадений → ищем везде
        collection = winners[0] if best > 0 and len(winners) == 1 else None
        reason = f"keyword scores: {scores}" if scores else "no collections"
        return Route(collection=collection, tools=suggest_tools(query), reason=reason)


class LLMRouter:
    def __init__(
        self,
        llm: LLM,
        fallback: Router | None = None,
        descriptions: Mapping[str, str] | None = None,
    ) -> None:
        self.llm = llm
        self.fallback = fallback or KeywordRouter()
        self.descriptions = dict(descriptions or DEFAULT_DESCRIPTIONS)

    def _prompt(self, query: str, collections: Sequence[str]) -> str:
        lines = [f"- {c}: {self.descriptions.get(c, '')}".rstrip(": ") for c in collections]
        return (
            "Choose where to search for the answer to the user query.\n"
            f"Collections:\n{chr(10).join(lines)}\n- all: search every collection\n"
            "Tools: calculator (arithmetic), get_date (current date/weekday).\n"
            f"<query>{query}</query>\n"
            'Return JSON: {"collection": "...", "tools": [...], "reason": "..."}'
        )

    def route(self, query: str, collections: Sequence[str]) -> Route:
        try:
            raw = self.llm.complete(
                self._prompt(query, collections),
                system="You are a query router for a retrieval system.",
                json_schema=ROUTE_SCHEMA,
            )
            if isinstance(raw, str):
                raw = json.loads(raw)
            decision = RouteDecision.model_validate(raw)
        except (ValidationError, ValueError, TypeError) as exc:
            log.info("LLM-роутер вернул невалидный ответ (%s) → keyword fallback", exc)
            return self._fallback(query, collections, "invalid LLM output")
        coll = decision.collection.strip().lower()
        if coll not in {c.lower() for c in collections} and coll not in {ALL, "none", ""}:
            return self._fallback(query, collections, f"unknown collection {coll!r}")
        collection = next((c for c in collections if c.lower() == coll), None)
        tools = [t for t in decision.tools if t in KNOWN_TOOLS]
        return Route(collection=collection, tools=tools, reason=decision.reason, router="llm")

    def _fallback(self, query: str, collections: Sequence[str], why: str) -> Route:
        r = self.fallback.route(query, collections)
        return r.model_copy(update={"router": "keyword-fallback", "reason": f"{why}; {r.reason}"})
