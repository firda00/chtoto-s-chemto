"""Правила/регэкспы признаков мошенничества (RU / UZ латиница+кириллица / EN) и ценовые признаки.

Каждый детектор возвращает `SignalHit`: число срабатываний и найденные фрагменты —
они идут и в модель (как числа), и в человекочитаемые причины отчёта.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer

from .config import districts_csv, uzs_per_usd

# --------------------------------------------------------------------------- нормализация

_APOSTROPHES = str.maketrans({"ʻ": "'", "ʼ": "'", "’": "'", "‘": "'", "`": "'", "ё": "е"})


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().translate(_APOSTROPHES)).strip()


# --------------------------------------------------------------------------- словари

URGENCY = [
    r"срочн\w*",
    r"только сегодня",
    r"до вечера",
    r"успейте",
    r"торопитесь",
    r"кто первый",
    r"последний шанс",
    r"горящ\w*",
    r"shoshilinch",
    r"tezkor",
    r"faqat bugun",
    r"bugunoq",
    r"шошилинч",
    r"тезкор",
    r"фақат бугун",
    r"\burgent\w*",
    r"today only",
    r"\bhurry\b",
    r"first come",
    r"\basap\b",
    r"won'?t last",
]
PREPAYMENT = [
    r"предоплат\w*",
    r"задат\w*",
    r"залог до просмотра",
    r"оплат\w* брон\w*",
    r"переведите",
    r"перевод\w* на карт\w*",
    r"скин\w* на карт\w*",
    r"до просмотра.{0,30}(?:оплат|перевод|залог)",
    r"oldindan to'?lov\w*",
    r"\bavans\w*",
    r"zakalat",
    r"kartaga o'?tkaz\w*",
    r"pul o'?tkaz\w*",
    r"олдиндан тўлов",
    r"аванс\w*",
    r"deposit before",
    r"advance payment",
    r"pay (?:the |a )?deposit",
    r"wire (?:the )?money",
    r"western union",
    r"send (?:the |me )?money",
    r"transfer .{0,20}(?:deposit|money|to my card)",
]
OWNER_ABROAD = [
    r"за границ\w*",
    r"за рубеж\w*",
    r"(?:нахожусь|живу|сейчас|уехал\w*)\s.{0,15}(?:в москв\w*|в корее|в турции|в европ\w*|в сша|"
    r"в дубае|в росси\w*)",
    r"ключи .{0,30}(?:курьер\w*|почт\w*|отправ\w*|вышлю)",
    r"показать не смогу",
    r"не могу показать",
    r"chet ?el\w*",
    r"xorij\w*",
    r"rossiyada\w*",
    r"kalit\w* .{0,25}(?:pochta|kuryer|yuboraman)",
    r"чет элда",
    r"хорижда",
    r"\babroad\b",
    r"out of the country",
    r"keys? .{0,25}(?:by mail|by courier|courier|by post)",
    r"can(?:'?t|not) show",
]
OFF_PLATFORM = [
    r"только (?:в |через )?(?:telegram|телеграм\w*|whatsapp|ватсап\w*|вотсап\w*)",
    r"пишите (?:в |на )?(?:telegram|телеграм\w*|whatsapp|ватсап\w*|вотсап\w*)",
    r"связь только\s.{0,12}(?:telegram|телеграм\w*|whatsapp|ватсап\w*)",
    r"звонки не принима\w*",
    r"не звоните",
    r"faqat (?:telegram\w*|whatsapp\w*)",
    r"telegramga yozing",
    r"qo'ng'iroq qilmang",
    r"only (?:on |via )?(?:telegram|whatsapp)",
    r"(?:telegram|whatsapp) only",
    r"(?:text|message|contact) me (?:on|via) (?:telegram|whatsapp)",
    r"no calls",
    r"t\.me/\w+",
    r"wa\.me/\d+",
    r"(?<![\w.])@[a-z][a-z0-9_]{4,}",
]
GENERIC_PHRASES = [
    r"отличная квартира",
    r"хорошая квартира",
    r"вс[её] есть",
    r"вс[её] необходимое",
    r"заходи и живи",
    r"для комфортной жизни",
    r"hammasi bor",
    r"zo'r kvartira",
    r"yaxshi kvartira",
    r"zo'r uy",
    r"great (?:apartment|flat)",
    r"nice (?:apartment|flat)",
    r"everything you need",
    r"perfect place",
]
# Конкретика: этаж, метро, ориентир, площадь — у мошеннических текстов её обычно мало.
SPECIFICS = [
    r"этаж",
    r"метро",
    r"ориентир",
    r"м²",
    r"кв\.? ?м",
    r"qavat",
    r"metro",
    r"mo'ljal",
    r"\bm2\b",
    r"m²",
    r"floor",
    r"landmark",
    r"station",
]
CARD_NUMBER = re.compile(r"\b(?:8600|9860|5614|4\d{3}|5\d{3})(?:[ -]?\d{4}){3}\b")


def _compile(patterns: list[str]) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{p})" for p in patterns))


_URGENCY = _compile(URGENCY)
_PREPAYMENT = _compile(PREPAYMENT)
_OWNER_ABROAD = _compile(OWNER_ABROAD)
_OFF_PLATFORM = _compile(OFF_PLATFORM)
_GENERIC = _compile(GENERIC_PHRASES)
_SPECIFICS = _compile(SPECIFICS)


@dataclass
class SignalHit:
    count: int = 0
    fragments: list[str] = field(default_factory=list)


def _find(pattern: re.Pattern[str], text: str) -> SignalHit:
    found = [m.group(0) for m in pattern.finditer(normalize_text(text))]
    return SignalHit(len(found), list(dict.fromkeys(found)))


def urgency(text: str) -> SignalHit:
    """Давление срочностью: «срочно», «только сегодня», «shoshilinch», «urgent»."""
    return _find(_URGENCY, text)


def prepayment(text: str) -> SignalHit:
    """Просьба о предоплате / задатке / переводе денег до просмотра."""
    return _find(_PREPAYMENT, text)


def owner_abroad(text: str) -> SignalHit:
    """Легенда «хозяин за границей, ключи пришлю курьером»."""
    return _find(_OWNER_ABROAD, text)


def off_platform_contact(text: str) -> SignalHit:
    """Уводит общение с площадки: только Telegram/WhatsApp, «звонки не принимаю»."""
    return _find(_OFF_PLATFORM, text)


def card_number(text: str) -> SignalHit:
    """Номер банковской карты (Uzcard 8600, Humo 9860, Visa/MC) прямо в тексте."""
    found = CARD_NUMBER.findall(text)
    return SignalHit(len(found), found)


def generic_description(text: str) -> tuple[float, SignalHit]:
    """Шаблонность описания 0..1: мало слов, нет конкретики, общие фразы."""
    norm = normalize_text(text)
    words = re.findall(r"\w+", norm)
    phrases = _find(_GENERIC, text)
    no_specifics = not _SPECIFICS.search(norm) and not re.search(r"\d", norm)
    score = (float(len(words) < 25) + float(no_specifics) + float(phrases.count > 0)) / 3
    return score, phrases


# --------------------------------------------------------------------------- цены по районам


def normalize_district(name: str) -> str:
    s = normalize_text(name)
    s = re.sub(r"\b(?:район|р-н|tumani|tuman|district)\b", "", s)
    return re.sub(r"[\s'\-.]", "", s)


@dataclass
class DistrictStats:
    district: str
    name_ru: str
    median_usd_m2: float
    std_usd_m2: float


class DistrictTable:
    """Медиана аренды $/м²/мес по районам Ташкента (справочник `data/district_prices.csv`)."""

    def __init__(self, rows: list[DistrictStats], aliases: dict[str, str]) -> None:
        self.rows = {r.district: r for r in rows}
        self._aliases = aliases
        medians = np.array([r.median_usd_m2 for r in rows])
        stds = np.array([r.std_usd_m2 for r in rows])
        self.city = DistrictStats(
            "Tashkent", "Ташкент", float(np.median(medians)), float(np.mean(stds))
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> DistrictTable:
        df = pd.read_csv(path or districts_csv())
        rows, aliases = [], {}
        for r in df.itertuples(index=False):
            rows.append(
                DistrictStats(r.district, r.name_ru, float(r.median_usd_m2), float(r.std_usd_m2))
            )
            for name in [r.district, r.name_ru, *str(r.aliases).split("|")]:
                if name and name != "nan":
                    aliases[normalize_district(name)] = r.district
        return cls(rows, aliases)

    def canonical(self, district: str) -> str | None:
        return self._aliases.get(normalize_district(district))

    def lookup(self, district: str) -> tuple[DistrictStats, bool]:
        """(статистика, известен ли район). Неизвестный район → средние по городу."""
        key = self.canonical(district)
        if key is None:
            return self.city, False
        return self.rows[key], True

    @property
    def districts(self) -> list[str]:
        return list(self.rows)


_UZS = {"UZS", "SUM", "SOM", "SO'M", "СУМ", "СЎМ"}


def to_usd(price: float, currency: str) -> float:
    if currency.strip().upper() in _UZS:
        return price / uzs_per_usd()
    return price


@dataclass
class PriceDeviation:
    price_usd: float
    usd_per_m2: float
    median_usd_m2: float
    ratio: float  # цена за м² / медиана района
    z: float  # (цена за м² − медиана) / std района
    district_known: bool
    district_name: str

    @property
    def too_good(self) -> bool:
        return self.ratio < 0.65


def price_deviation(
    price: float, currency: str, area_m2: float, district: str, table: DistrictTable
) -> PriceDeviation:
    stats, known = table.lookup(district)
    usd = to_usd(price, currency)
    per_m2 = usd / max(area_m2, 1.0)
    return PriceDeviation(
        price_usd=usd,
        usd_per_m2=per_m2,
        median_usd_m2=stats.median_usd_m2,
        ratio=per_m2 / stats.median_usd_m2,
        z=(per_m2 - stats.median_usd_m2) / stats.std_usd_m2,
        district_known=known,
        district_name=stats.name_ru,
    )


# --------------------------------------------------------------------------- корпусные признаки


def shingle_vectorizer() -> HashingVectorizer:
    """Словесные 3-граммы (шинглы), бинарно, без знака — Жаккар через скалярное произведение."""
    return HashingVectorizer(
        analyzer="word",
        ngram_range=(3, 3),
        n_features=2**20,
        alternate_sign=False,
        norm=None,
        binary=True,
        token_pattern=r"(?u)\b\w+\b",
        preprocessor=normalize_text,
    )


class TextIndex:
    """Поиск дублированного текста: max Jaccard по шинглам с другими объявлениями корпуса."""

    def __init__(self) -> None:
        self._vec = shingle_vectorizer()
        self.ids: list[str] = []
        self._pos: dict[str, list[int]] = {}
        self._X = None
        self._XT = None
        self._sizes = np.zeros(0)

    def fit(self, ids: list[str], texts: list[str]) -> TextIndex:
        self.ids = list(ids)
        self._pos: dict[str, list[int]] = defaultdict(list)
        for i, x in enumerate(self.ids):
            self._pos[x].append(i)
        if texts:
            self._X = self._vec.transform(texts).tocsr()
            self._XT = self._X.T.tocsr()  # (признаки × документы): q @ XT — быстрый csr@csr
            self._sizes = np.asarray(self._X.sum(axis=1)).ravel()
        return self

    def max_jaccard(self, text: str, exclude_id: str | None = None) -> tuple[float, str | None]:
        if self._X is None:
            return 0.0, None
        q = self._vec.transform([text])
        q_size = q.sum()
        if q_size == 0:
            return 0.0, None
        inter = (q @ self._XT).toarray().astype(float).ravel()
        union = self._sizes + q_size - inter
        jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        if exclude_id is not None:
            jac[self._pos.get(exclude_id, [])] = 0.0
        best = int(np.argmax(jac))
        return float(jac[best]), (self.ids[best] if jac[best] > 0 else None)


def normalize_phone(phone: str) -> str:
    digits = re.sub(r"\D", "", phone)
    return digits[-9:]  # +998 XX XXX XX XX → последние 9 цифр


class PhoneRegistry:
    """Сколько других объявлений используют тот же телефон."""

    def __init__(self) -> None:
        self._by_phone: dict[str, set[str]] = defaultdict(set)

    def add(self, listing_id: str, phone: str) -> None:
        key = normalize_phone(phone)
        if key:
            self._by_phone[key].add(listing_id)

    def reuse_count(self, phone: str, exclude_id: str | None = None) -> int:
        key = normalize_phone(phone)
        if not key:
            return 0
        return len(self._by_phone.get(key, set()) - {exclude_id})
