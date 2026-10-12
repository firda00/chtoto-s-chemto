"""Общая лёгкая обработка текста: токенизация, стоп-слова, грубый стемминг RU/EN.

Используется BM25, HashingEmbedder, лексическим реранкером, роутером и MockLLM —
чтобы все компоненты одинаково «понимали» слова без тяжёлых NLP-зависимостей.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"\w+", re.UNICODE)

_STOP_EN = (
    "a an the is are was were be been of to in on at for and or but with by from as it its this "
    "that these those i you we they he she do does did can could how what which who whom when "
    "where why much many my your our there here have has had will would should please about "
    "into per"
)
_STOP_RU = (
    "и в во на с со к ко по о об от до за из у же ли не ни а но или что как какой какая какие "
    "каком какое сколько скольки где когда кто чем это этот эта эти то так для при мне меня мы вы "
    "он она они его её их есть был была было быть бы ещё уже очень можно нужно надо"
)
STOPWORDS: frozenset[str] = frozenset((_STOP_EN + " " + _STOP_RU).split())

_RU_ENDINGS_RAW = (
    "ами ями ого его ому ему ыми ими ов ев ей ой ий ый ая яя ое ее ые ие ую юю ам ям ах ях "
    "ом ем ть ет ит ут ют ат ят а я о е ы и у ю ь"
)
# Окончания отсортированы по убыванию длины — отрезаем самое длинное подходящее.
_RU_ENDINGS = tuple(
    sorted(
        _RU_ENDINGS_RAW.split(),
        key=len,
        reverse=True,
    )
)
_EN_ENDINGS = ("ing", "ies", "ed", "es", "s")
_CYR = re.compile(r"[а-яё]")


def stem(token: str) -> str:
    """Очень грубый стеммер: срезает частые окончания и обрезает до 7 символов."""
    if token.isdigit() or len(token) <= 3:
        return token
    if _CYR.search(token):
        for end in _RU_ENDINGS:
            if token.endswith(end) and len(token) - len(end) >= 3:
                token = token[: -len(end)]
                break
    else:
        for end in _EN_ENDINGS:
            if token.endswith(end) and len(token) - len(end) >= 3:
                token = token[: -len(end)]
                break
    return token[:7]


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def tokenize(text: str, *, use_stem: bool = True, drop_stopwords: bool = True) -> list[str]:
    out: list[str] = []
    for w in words(text):
        if drop_stopwords and w in STOPWORDS:
            continue
        out.append(stem(w) if use_stem else w)
    return out


def normalize_query(text: str) -> str:
    """Препроцессинг запроса: схлопывает пробелы, убирает мусорную пунктуацию по краям."""
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" \t\"'`«»")
