"""Разбор текста вакансии в структуру ``Vacancy``.

1. Rule-based: заголовки блоков («Требования», «Будет плюсом», «Requirements», «Nice to have»),
   словарь навыков с синонимами, регулярки для опыта («от 3 лет», «5+ years»).
2. Опционально LLM structured extraction по JSON Schema модели ``Vacancy``;
   результат LLM объединяется с правилами (правила — страховка от пропусков и галлюцинаций).
"""

from __future__ import annotations

import logging
import re

from pydantic import ValidationError

from .llm import LLM, MockLLM, embed_input
from .models import Vacancy
from .parsers import detect_language
from .skills import canonicalize, contains_keyword, find_skills, find_tech_tokens

log = logging.getLogger(__name__)

MUST_HEADINGS = {
    "требования",
    "требования к кандидату",
    "основные требования",
    "обязательные требования",
    "ключевые требования",
    "мы ожидаем",
    "что мы ждем",
    "что мы ждём",
    "от тебя",
    "от вас",
    "необходимо",
    "обязательно",
    "нам важно",
    "наши ожидания",
    "requirements",
    "required",
    "must have",
    "must-have",
    "what you'll need",
    "what you will need",
    "what we're looking for",
    "what we are looking for",
    "qualifications",
    "required qualifications",
    "minimum qualifications",
    "you have",
    "about you",
    "who you are",
}
NICE_HEADINGS = {
    "будет плюсом",
    "плюсом будет",
    "будет преимуществом",
    "преимуществом будет",
    "желательно",
    "дополнительно",
    "nice to have",
    "nice-to-have",
    "bonus",
    "bonus points",
    "preferred",
    "preferred qualifications",
    "pluses",
    "good to have",
    "it's a plus",
    "a plus",
}
OTHER_HEADINGS = {
    "обязанности",
    "задачи",
    "чем предстоит заниматься",
    "что нужно делать",
    "чем заниматься",
    "условия",
    "мы предлагаем",
    "что мы предлагаем",
    "о компании",
    "о нас",
    "о проекте",
    "responsibilities",
    "what you'll do",
    "what you will do",
    "about us",
    "about the role",
    "about the company",
    "the role",
    "we offer",
    "what we offer",
    "benefits",
    "perks",
}

_INLINE_NICE = re.compile(
    r"плюсом|преимуществ|желательно|nice[ -]to[ -]have|\ba plus\b|\bpreferred\b|\bbonus\b",
    re.I,
)

_TITLE_PREFIX = re.compile(
    r"^(?:вакансия|позиция|должность|position|job title|role|title)\s*[:\-—]\s*", re.I
)

_YEARS = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*\+?\s*(?:(?:-|–|—|до|to)\s*\d+(?:[.,]\d+)?\s*)?\+?\s*"
    r"(?:лет|года|год|years?|yrs)",
    re.I,
)


def _norm(line: str) -> str:
    s = re.sub(r"^#{1,6}\s*", "", line.strip())
    s = s.strip("*_•-–—: \t").rstrip(":").strip().lower()
    return re.sub(r"\s+", " ", s)


def _block_kind(line: str) -> str | None:
    if len(line.strip()) > 70:
        return None
    key = _norm(line)
    if key in MUST_HEADINGS:
        return "must"
    if key in NICE_HEADINGS:
        return "nice"
    if key in OTHER_HEADINGS:
        return "other"
    return None


def split_blocks(text: str) -> dict[str, list[str]]:
    """Разбить вакансию на блоки: header/must/nice/other (списки строк)."""
    blocks: dict[str, list[str]] = {"header": [], "must": [], "nice": [], "other": []}
    current = "header"
    for line in text.splitlines():
        kind = _block_kind(line)
        if kind:
            current = kind
            continue
        if line.strip():
            blocks[current].append(line.strip())
    return blocks


def extract_title(text: str) -> str:
    for line in text.splitlines():
        s = re.sub(r"^#{1,6}\s*", "", line.strip()).strip(" *")
        m = _TITLE_PREFIX.match(s)
        if m:
            return s[m.end() :].strip(" *")
    for line in text.splitlines():
        s = re.sub(r"^#{1,6}\s*", "", line.strip()).strip(" *")
        if s and _block_kind(s) is None:
            return s[:120]
    return ""


def extract_min_years(lines: list[str]) -> float | None:
    """Минимальный опыт из строк вида «опыт от 3 лет», «3+ years of experience»."""
    for line in lines:
        low = line.lower()
        if not re.search(r"опыт|experience|стаж", low):
            continue
        m = _YEARS.search(line)
        if m:
            return float(m.group(1).replace(",", "."))
    return None


_ALT_SPLIT = re.compile(r"[,;()\[\]]")
_ALT_WORD = re.compile(r"\s(?:или|либо|or)\s|\s/\s", re.I)


def find_alternatives(line: str) -> list[list[str]]:
    """Группы взаимозаменяемых навыков: «PyTorch или TensorFlow», «AWS or GCP»."""
    groups: list[list[str]] = []
    for chunk in _ALT_SPLIT.split(line):
        if not _ALT_WORD.search(chunk):
            continue
        skills = _ordered(find_skills(chunk) | find_tech_tokens(chunk), chunk)
        if len(skills) >= 2:
            groups.append(skills)
    return groups


def _ordered(items: set[str], text: str) -> list[str]:
    """Отсортировать навыки по первому упоминанию в тексте (стабильно)."""
    low = text.lower()

    def pos(item: str) -> tuple[int, str]:
        idx = low.find(item.lower())
        return (idx if idx >= 0 else 10**9, item)

    return sorted(items, key=pos)


def extract_vacancy_rules(text: str) -> Vacancy:
    blocks = split_blocks(text)
    structured = bool(blocks["must"] or blocks["nice"])

    must: set[str] = set()
    nice: set[str] = set()
    alternatives: list[list[str]] = []
    for line in blocks["must"] + blocks["nice"] if structured else text.splitlines():
        for group in find_alternatives(line):
            if group not in alternatives:
                alternatives.append(group)
    if structured:
        for line in blocks["must"]:
            skills = find_skills(line) | find_tech_tokens(line)
            (nice if _INLINE_NICE.search(line) else must).update(skills)
        for line in blocks["nice"]:
            nice.update(find_skills(line) | find_tech_tokens(line))
    else:
        # Неструктурированный текст: строки с «плюсом» -> nice, остальные навыки -> must.
        for line in text.splitlines():
            skills = find_skills(line)
            (nice if _INLINE_NICE.search(line) else must).update(skills)
    nice -= must

    all_skills = find_skills(text)
    keywords = must | nice | all_skills
    min_years = extract_min_years(blocks["must"]) or extract_min_years(text.splitlines())
    return Vacancy(
        title=extract_title(text),
        must_have=_ordered(must, text),
        nice_to_have=_ordered(nice, text),
        min_years=min_years,
        keywords=_ordered(keywords, text),
        alternatives=[g for g in alternatives if set(g) <= must or set(g) <= nice],
        language=detect_language(text),  # type: ignore[arg-type]
        raw_text=text,
    )


VACANCY_SYSTEM = (
    "Ты — HR-аналитик. Извлеки из текста вакансии структуру: название позиции, "
    "обязательные навыки (must_have), желательные (nice_to_have), минимальный опыт в годах "
    "(min_years) и ключевые слова. Навыки пиши короткими каноническими названиями "
    "(Python, Docker, LLM, RAG). Ничего не придумывай: только то, что есть в тексте."
)


def _llm_schema() -> dict:
    schema = Vacancy.model_json_schema()
    for field in ("raw_text", "alternatives", "language"):
        schema.get("properties", {}).pop(field, None)
    return schema


def _clean_list(items: object) -> list[str]:
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for it in items:
        if isinstance(it, str) and it.strip():
            c = canonicalize(it.strip())
            if c not in out:
                out.append(c)
    return out


def extract_vacancy_llm(text: str, llm: LLM) -> Vacancy | None:
    prompt = "Извлеки структуру вакансии. Ответ — JSON по схеме.\n\n" + embed_input(
        {"task": "extract_vacancy", "text": text}
    )
    try:
        raw = llm.complete(prompt, system=VACANCY_SYSTEM, json_schema=_llm_schema())
        if not isinstance(raw, dict) or not raw:
            return None
        data = {
            "title": str(raw.get("title") or ""),
            "must_have": _clean_list(raw.get("must_have")),
            "nice_to_have": _clean_list(raw.get("nice_to_have")),
            "min_years": raw.get("min_years"),
            "keywords": _clean_list(raw.get("keywords")),
        }
        return Vacancy.model_validate(data)
    except (ValidationError, ValueError, TypeError) as exc:
        log.warning("LLM-извлечение вакансии не удалось: %s", exc)
        return None
    except Exception as exc:  # сеть/провайдер — не валим весь пайплайн
        log.warning("Ошибка LLM при извлечении вакансии: %s", exc)
        return None


def _grounded(items: list[str], text: str) -> list[str]:
    """Оставить только навыки, реально упомянутые в тексте (защита от галлюцинаций LLM)."""
    return [i for i in items if contains_keyword(text, i)]


def merge_vacancies(rules: Vacancy, llm_v: Vacancy) -> Vacancy:
    text = rules.raw_text
    must = list(dict.fromkeys(rules.must_have + _grounded(llm_v.must_have, text)))
    nice = [
        s
        for s in dict.fromkeys(rules.nice_to_have + _grounded(llm_v.nice_to_have, text))
        if s not in must
    ]
    keywords = list(dict.fromkeys(rules.keywords + must + nice + _grounded(llm_v.keywords, text)))
    return Vacancy(
        title=llm_v.title or rules.title,
        must_have=must,
        nice_to_have=nice,
        min_years=llm_v.min_years if llm_v.min_years is not None else rules.min_years,
        keywords=keywords,
        alternatives=rules.alternatives,
        language=rules.language,
        raw_text=text,
    )


def extract_vacancy(text: str, llm: LLM | None = None) -> Vacancy:
    """Правила всегда; LLM — если передана и это не пустой мок."""
    rules = extract_vacancy_rules(text)
    if llm is None:
        return rules
    llm_v = extract_vacancy_llm(text, llm)
    if llm_v is None:
        return rules
    return merge_vacancies(rules, llm_v)


def is_real_llm(llm: LLM | None) -> bool:
    return llm is not None and not isinstance(llm, MockLLM)
