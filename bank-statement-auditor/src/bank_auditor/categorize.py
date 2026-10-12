"""Категоризация трат: правила по ключевым словам (RU/EN/UZ) + опциональный LLM-фолбэк."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .llm import LLM
from .models import CATEGORIES, Transaction
from .normalize import translit_ru_to_lat

# Ключевые слова. Короткие (< 5 символов) совпадают только целым словом,
# длинные — как начало слова («APTEK» ловит «APTEKA», «АПТЕКА»).
KEYWORDS: dict[str, tuple[str, ...]] = {
    "food": (
        "KORZINKA",
        "MAKRO",
        "HAVAS",
        "MAGNUM",
        "CARREFOUR",
        "BARAKA",
        "EVOS",
        "KFC",
        "OQTEPA",
        "LAVASH",
        "SAFIA",
        "BELLISSIMO",
        "MAX WAY",
        "MAXWAY",
        "CHOPAR",
        "FEED UP",
        "DODO",
        "PIZZA",
        "BURGER",
        "CAFE",
        "KAFE",
        "COFFEE HOUSE",
        "COFFEE",
        "KOFE",
        "RESTORAN",
        "RESTAURANT",
        "CHOYXONA",
        "OSHXONA",
        "SUPERMARKET",
        "PRODUKT",
        "ПРОДУКТ",
        "КАФЕ",
        "РЕСТОРАН",
        "YANDEX EDA",
        "YANDEX EATS",
        "WOLT",
        "EXPRESS24",
        "UZUM TEZKOR",
        "STARBUCKS",
        "BAKERY",
        "NONVOYXONA",
        "GROCERY",
        "FOOD",
    ),
    "transport": (
        "YANDEX GO",
        "YANDEX TAXI",
        "TAXI",
        "ТАКСИ",
        "MYTAXI",
        "UBER",
        "METRO",
        "МЕТРО",
        "AVTOBUS",
        "АВТОБУС",
        "AZS",
        "АЗС",
        "BENZIN",
        "БЕНЗИН",
        "UZBEKISTAN AIRWAYS",
        "UZBEKISTAN TEMIR",
        "AFROSIYOB",
        "PARKING",
        "ПАРКОВК",
        "METAN",
        "PROPAN",
        "FUEL",
    ),
    "subscriptions": (
        "NETFLIX",
        "SPOTIFY",
        "APPLE BILL",
        "APPLE COM BILL",
        "ICLOUD",
        "YOUTUBE PREMIUM",
        "YOUTUBEPREMIUM",
        "GOOGLE ONE",
        "GOOGLE STORAGE",
        "YANDEX PLUS",
        "KINOPOISK",
        "MEGOGO",
        "IVI",
        "OKKO",
        "AMEDIATEKA",
        "CHATGPT",
        "OPENAI",
        "ANTHROPIC",
        "ADOBE",
        "CANVA",
        "MICROSOFT",
        "OFFICE 365",
        "NOTION",
        "DROPBOX",
        "TELEGRAM PREMIUM",
        "PODPISKA",
        "ПОДПИСКА",
        "OBUNA",
        "SUBSCRIPTION",
        "DISNEY",
    ),
    "utilities": (
        "BEELINE",
        "UCELL",
        "UZMOBILE",
        "MOBIUZ",
        "UMS",
        "HUDUDGAZ",
        "HUDUDIY ELEKTR",
        "ELEKTR",
        "TOSHSHAHARSUV",
        "SUVSOZ",
        "SUVOQOVA",
        "KOMMUNAL",
        "КОММУНАЛ",
        "ЭЛЕКТРОЭНЕРГ",
        "UZTELECOM",
        "TURON TELECOM",
        "SARKOR",
        "INTERNET",
        "ИНТЕРНЕТ",
        "ISSIQLIK",
        "CHIQINDI",
        "UTILITY",
        "UTILITIES",
    ),
    "health": (
        "APTEKA",
        "АПТЕКА",
        "DORIXONA",
        "PHARM",
        "CLINIC",
        "KLINIKA",
        "КЛИНИК",
        "STOMATOLOG",
        "СТОМАТОЛОГ",
        "FITNESS",
        "ФИТНЕС",
        "GYM",
        "SPORT CLUB",
        "SPORTZAL",
        "MEDICAL",
        "MED CENTER",
        "LABORATORIYA",
        "SHIFOXONA",
        "HOSPITAL",
        "DENTAL",
    ),
    "entertainment": (
        "KINOTEATR",
        "CINEMA",
        "КИНОТЕАТР",
        "MAGIC CITY",
        "STEAM",
        "PLAYSTATION",
        "BOWLING",
        "BOULING",
        "KONSERT",
        "CONCERT",
        "TEATR",
        "ТЕАТР",
        "ICE CITY",
        "KARTING",
        "QUEST",
        "CINEMAX",
    ),
    "shopping": (
        "UZUM MARKET",
        "UZUM",
        "TEXNOMART",
        "MEDIAPARK",
        "ARTEL",
        "ZARA",
        "LC WAIKIKI",
        "DEFACTO",
        "WILDBERRIES",
        "OZON",
        "ALIEXPRESS",
        "IKEA",
        "MALL",
        "BOZOR",
        "RYNOK",
        "РЫНОК",
        "MAGAZIN",
        "МАГАЗИН",
        "ELMAKON",
        "GALMART",
        "AMAZON",
    ),
    "transfers": (
        "PEREVOD",
        "ПЕРЕВОД",
        "P2P",
        "TRANSFER",
        "O'TKAZMA",
        "OTKAZMA",
        "CARD TO CARD",
        "CARD2CARD",
        "KARTADAN KARTAGA",
    ),
    "cash": (
        "ATM",
        "БАНКОМАТ",
        "BANKOMAT",
        "СНЯТИЕ НАЛИЧН",
        "SNYATIE NALICHN",
        "NAQD",
        "CASH WITHDRAWAL",
        "CASH",
    ),
    "salary": (
        "ЗАРПЛАТА",
        "ZARPLATA",
        "SALARY",
        "ISH HAQI",
        "OYLIK",
        "PAYROLL",
        "АВАНС",
        "AVANS",
    ),
}

CategoryName = Literal[
    "food",
    "transport",
    "subscriptions",
    "utilities",
    "health",
    "entertainment",
    "shopping",
    "transfers",
    "cash",
    "salary",
    "other",
]


class CategoryGuess(BaseModel):
    """Structured output от LLM для неизвестного мерчанта."""

    category: CategoryName
    confidence: float = Field(ge=0, le=1)
    reason: str = ""


def _compile(keyword: str) -> re.Pattern[str]:
    kw = re.escape(keyword)
    if len(keyword) < 5:
        return re.compile(rf"(?<![\w']){kw}(?![\w'])")
    return re.compile(rf"(?<![\w']){kw}")


_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    (cat, kw, _compile(kw)) for cat, kws in KEYWORDS.items() for kw in kws
]

LLM_SYSTEM = (
    "Ты помогаешь категоризировать банковские операции. Ответь строго JSON по схеме. "
    f"Допустимые категории: {', '.join(CATEGORIES)}."
)


def _haystack(t: Transaction) -> str:
    desc = re.sub(r"[^\w' ]+", " ", t.description.upper())
    return f" {t.merchant_normalized} | {translit_ru_to_lat(desc)} | {desc} "


def rule_category(t: Transaction) -> str | None:
    """Категория по правилам или None. Побеждает самое длинное совпавшее ключевое слово."""
    hay = _haystack(t)
    best: tuple[int, str] | None = None
    for cat, kw, pattern in _PATTERNS:
        if t.direction == "credit" and cat not in {"salary", "transfers", "cash"}:
            continue
        if t.direction == "debit" and cat == "salary":
            continue
        if pattern.search(hay) and (best is None or len(kw) > best[0]):
            best = (len(kw), cat)
    if best is not None:
        return best[1]
    if t.direction == "credit":
        return "transfers" if t.amount > 0 else "other"
    return None


class Categorizer:
    """Правила → (опционально) LLM для мерчантов, которых правила не узнали."""

    def __init__(self, llm: LLM | None = None, min_confidence: float = 0.5) -> None:
        self.llm = llm
        self.min_confidence = min_confidence
        self.cache: dict[str, str] = {}
        self.llm_calls = 0
        self.llm_errors = 0

    def categorize_one(self, t: Transaction) -> str:
        cat = rule_category(t)
        if cat is not None:
            return cat
        if self.llm is None:
            return "other"
        key = t.merchant_normalized
        if key not in self.cache:
            self.cache[key] = self._ask_llm(t)
        return self.cache[key]

    def _ask_llm(self, t: Transaction) -> str:
        self.llm_calls += 1
        prompt = (
            "Определи категорию траты по описанию банковской операции.\n"
            f"Мерчант: {t.merchant_normalized}\nОписание: {t.description}\n"
            f"Сумма: {t.amount:.2f} {t.currency}"
        )
        try:
            raw = self.llm.complete(
                prompt, system=LLM_SYSTEM, json_schema=CategoryGuess.model_json_schema()
            )
            guess = CategoryGuess.model_validate(raw)
        except (ValidationError, ValueError, TypeError):
            self.llm_errors += 1
            return "other"
        except Exception:  # noqa: BLE001 — сеть/провайдер: не роняем аудит
            self.llm_errors += 1
            return "other"
        return guess.category if guess.confidence >= self.min_confidence else "other"

    def categorize(self, txns: list[Transaction]) -> list[Transaction]:
        return [t.model_copy(update={"category": self.categorize_one(t)}) for t in txns]


def categorize(txns: list[Transaction], llm: LLM | None = None) -> list[Transaction]:
    return Categorizer(llm=llm).categorize(txns)
