"""Адаптация резюме под вакансию и платформу + сопроводительное письмо.

Вся генерация идёт через ``llm.complete``. Результат LLM проходит «guard»:
раздел откатывается к оригиналу, если в нём появились компании, числа/даты или навыки,
которых не было в исходном резюме. Так соблюдается правило «никогда не выдумывать опыт».
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import ValidationError

from .llm import LLM, embed_input, parse_json_response
from .models import (
    SECTION_NAMES,
    ChangeEntry,
    CoverLetter,
    LLMTailorOutput,
    Resume,
    TailoredResume,
    Vacancy,
)
from .parsers import parse_resume
from .scoring import score_resume
from .skills import SYNONYMS, contains_keyword, find_skills, variants

log = logging.getLogger(__name__)

PLATFORM_RULES: dict[str, dict[str, Any]] = {
    "linkedin": {
        "order": ["summary", "experience", "skills", "education", "contacts"],
        "rules": [
            "Раздел summary — это LinkedIn «About»: 3–5 коротких абзацев от первого лица, до 2600 символов.",
            "Первая фраза summary — сильный headline: роль + 2–3 ключевых навыка из вакансии.",
            "Опыт — маркированные пункты с глаголами действия и измеримым результатом (если он есть в оригинале).",
            "Skills — список отдельных навыков, самые релевантные вакансии первыми.",
        ],
    },
    "indeed": {
        "order": ["contacts", "summary", "skills", "experience", "education"],
        "rules": [
            "Простой текст без таблиц, колонок, эмодзи и графики.",
            "Стандартные названия разделов; ключевые слова вакансии в точном написании (если они правдивы).",
            "Summary — 2–4 предложения; опыт — маркированные пункты.",
        ],
    },
    "hh": {
        "order": ["contacts", "summary", "skills", "experience", "education"],
        "rules": [
            "Раздел «Ключевые навыки» — короткие теги через запятую (как поле навыков hh.ru).",
            "«О себе» — 3–5 предложений, без воды, с упором на требования вакансии.",
            "В опыте — обязанности и достижения маркированным списком; без таблиц.",
        ],
    },
    "generic": {
        "order": ["contacts", "summary", "skills", "experience", "education"],
        "rules": [
            "ATS-friendly: одна колонка, стандартные заголовки, маркированные списки.",
            "Ключевые слова вакансии — только там, где они подтверждаются опытом.",
        ],
    },
}

SECTION_TITLES: dict[str, dict[str, str]] = {
    "ru": {
        "contacts": "Контакты",
        "summary": "О себе",
        "skills": "Ключевые навыки",
        "experience": "Опыт работы",
        "education": "Образование",
        "projects": "Проекты",
        "languages": "Языки",
        "other": "Дополнительно",
    },
    "en": {
        "contacts": "Contact",
        "summary": "Summary",
        "skills": "Skills",
        "experience": "Experience",
        "education": "Education",
        "projects": "Projects",
        "languages": "Languages",
        "other": "Additional",
    },
}

TAILOR_SYSTEM = """Ты — карьерный редактор и эксперт по ATS. Адаптируй резюме под вакансию и платформу.

ЖЁСТКИЕ ПРАВИЛА:
1. Никогда не выдумывай опыт: компании, должности, даты, цифры, проекты, образование, навыки.
2. Разрешено только: перефразировать, переупорядочивать, сокращать, использовать формулировки
   вакансии для того, что УЖЕ есть в резюме, добавлять ключевые слова только из present_keywords.
3. Слова из missing_keywords НЕ добавлять ни в каком виде — у кандидата нет подтверждения.
4. Сохраняй язык исходного резюме и все факты (даты, названия компаний, цифры) дословно.
5. Верни JSON: {"sections": {<те же ключи разделов>: "<текст>"}, "changes": [{"section": ...,
   "action": "rephrased|reordered|keywords_added|unchanged|formatted", "detail": "что и зачем"}]}.
"""

COVER_SYSTEM = """Ты пишешь сопроводительное письмо кандидата. Правила:
1. Только факты из резюме: никаких выдуманных компаний, цифр, навыков и проектов.
2. Не заявляй навыки из missing_keywords.
3. Язык письма — как у вакансии (language). Тон — деловой, без клише и лести.
4. Объём и стиль — по правилам платформы. Верни только текст письма.
"""

COVER_RULES = {
    "linkedin": "Короткое сообщение рекрутеру (InMail): 80–150 слов, 3 абзаца.",
    "indeed": "Классическое письмо: 200–300 слов, приветствие, 3 абзаца, подпись.",
    "hh": "Сопроводительное для hh.ru: 100–180 слов, без формальностей, по делу.",
    "generic": "Письмо 150–250 слов: зачем откликаюсь, 2–3 релевантных достижения, призыв к диалогу.",
}


# ------------------------------------------------------------------ guard


_LEGAL_FORM = re.compile(
    r"(?:(?:ООО|АО|ПАО|ЗАО|ОАО|ИП|ГК|НКО)\s*[«\"']?[^«»\"'\n,;]{1,40}[»\"']?"
    r"|[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*){0,3}\s*,?\s+(?:LLC|Inc\.?|Ltd\.?|GmbH|Corp\.?|LLP|PLC|S\.A\.))"
)
_QUOTED = re.compile(r"«([^»\n]{2,40})»")
_AT_COMPANY = re.compile(
    r"(?:\bat|\bin|\bfor|\bjoined|\bв компании|\bв компанию|\bкомпании|\bкомпания)\s+"
    r"([A-ZА-ЯЁ][\w&.-]+(?:\s+[A-ZА-ЯЁ][\w&.-]+){0,2})"
)
_LATIN_PROPER = re.compile(r"(?<![\w.@/])([A-Z][a-zA-Z0-9&]+(?:\s+[A-Z][a-zA-Z0-9&]+){0,2})")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?%?")

_KNOWN_TERMS: set[str] = {v.lower() for c, vs in SYNONYMS.items() for v in [c, *vs]}


def extract_organizations(text: str) -> set[str]:
    """Грубое извлечение названий организаций (для проверки «не выдумали ли компанию»)."""
    orgs: set[str] = set()
    for m in _LEGAL_FORM.finditer(text):
        orgs.add(re.sub(r"\s+", " ", m.group(0)).strip(" ,.;"))
    for m in _QUOTED.finditer(text):
        orgs.add(m.group(1).strip())
    for m in _AT_COMPANY.finditer(text):
        orgs.add(m.group(1).strip(" ,.;"))
    if re.search(r"[А-Яа-яЁё]", text):
        # В русском тексте латиница с заглавной — почти всегда имя собственное или технология.
        for m in _LATIN_PROPER.finditer(text):
            orgs.add(m.group(1).strip())
    return {o for o in orgs if o and o.lower() not in _KNOWN_TERMS}


def _numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in _NUMBER.findall(text)}


def _has_unknown_word(name: str, base_low: str) -> bool:
    for w in name.split():
        w = w.strip("«»\"'.,;:()").lower()
        if len(w) < 2 or w in _KNOWN_TERMS or w in base_low:
            continue
        return True
    return False


def find_violations(original_resume_text: str, new_text: str, allowed_text: str = "") -> list[str]:
    """Что появилось в новом тексте и не подтверждается исходным резюме."""
    base = original_resume_text + "\n" + allowed_text
    base_low = base.lower()
    problems: list[str] = []

    new_skills = sorted(find_skills(new_text) - find_skills(original_resume_text))
    if new_skills:
        problems.append("новые навыки: " + ", ".join(new_skills))

    new_nums = sorted(_numbers(new_text) - _numbers(base))
    if new_nums:
        problems.append("новые числа/даты: " + ", ".join(new_nums))

    new_orgs = sorted(
        o
        for o in extract_organizations(new_text)
        if o.lower() not in base_low and _has_unknown_word(o, base_low)
    )
    if new_orgs:
        problems.append("новые организации/названия: " + ", ".join(new_orgs))
    return problems


# ------------------------------------------------------------------ helpers


def present_keywords(resume: Resume, vacancy: Vacancy) -> list[str]:
    """Ключевые слова вакансии, которые подтверждаются текстом резюме."""
    return [k for k in vacancy.keywords if contains_keyword(resume.raw_text, k)]


def _ordered_sections(sections: dict[str, str], platform: str) -> dict[str, str]:
    order = PLATFORM_RULES[platform]["order"]
    out = {k: sections[k] for k in order if k in sections}
    out.update({k: v for k, v in sections.items() if k not in out})
    return out


def resume_from_sections(sections: dict[str, str], language: str) -> Resume:
    titles = SECTION_TITLES.get(language, SECTION_TITLES["ru"])
    parts: list[str] = []
    for key, body in sections.items():
        if key == "contacts":
            parts.insert(0, body)
            continue
        parts.append(f"## {titles.get(key, key)}\n{body}")
    return parse_resume("\n\n".join(parts))


def _highlights(resume: Resume, limit: int = 3) -> list[str]:
    exp = resume.section("experience")
    bullets = [
        ln.strip().lstrip("-*• ").strip()
        for ln in exp.splitlines()
        if ln.strip().startswith(("-", "*", "•"))
    ]
    return bullets[:limit]


# ------------------------------------------------------------------ tailoring


def build_tailor_prompt(resume: Resume, vacancy: Vacancy, platform: str) -> str:
    present = present_keywords(resume, vacancy)
    in_skills = [k for k in present if contains_keyword(resume.section("skills"), k)]
    missing = [k for k in vacancy.keywords if k not in present]
    payload = {
        "task": "tailor_resume",
        "platform": platform,
        "platform_rules": PLATFORM_RULES[platform]["rules"],
        "language": resume.language,
        "vacancy": {
            "title": vacancy.title,
            "must_have": vacancy.must_have,
            "nice_to_have": vacancy.nice_to_have,
            "min_years": vacancy.min_years,
        },
        "present_keywords": present,
        "keywords_in_skills": in_skills,
        "missing_keywords": missing,
        "sections": {k: v for k, v in resume.sections.items()},
    }
    return (
        f"Адаптируй резюме под вакансию «{vacancy.title}» для платформы {platform}. "
        "Соблюдай правила платформы и жёсткие правила из системного сообщения.\n\n"
        + embed_input(payload)
    )


def tailor_resume(
    resume: Resume, vacancy: Vacancy, llm: LLM, platform: str = "generic"
) -> TailoredResume:
    if platform not in PLATFORM_RULES:
        raise ValueError(f"Неизвестная платформа: {platform}. Доступны: {sorted(PLATFORM_RULES)}")

    warnings: list[str] = []
    prompt = build_tailor_prompt(resume, vacancy, platform)
    try:
        raw = llm.complete(
            prompt, system=TAILOR_SYSTEM, json_schema=LLMTailorOutput.model_json_schema()
        )
        if isinstance(raw, str):
            raw = parse_json_response(raw)
        out = LLMTailorOutput.model_validate(raw)
    except (ValidationError, ValueError, TypeError) as exc:
        log.warning("Некорректный ответ LLM: %s", exc)
        warnings.append(
            f"LLM вернула некорректный ответ, резюме оставлено без изменений ({exc.__class__.__name__})"
        )
        out = LLMTailorOutput(sections=dict(resume.sections), changes=[])

    # Разрешённый «чужой» текст — только название позиции и подтверждённые ключевые слова.
    present = present_keywords(resume, vacancy)
    allowed = vacancy.title + "\n" + "\n".join(v for k in present for v in variants(k))

    final: dict[str, str] = {}
    changelog: list[ChangeEntry] = []
    reverted: set[str] = set()
    for key, original in resume.sections.items():
        new = out.sections.get(key)
        if new is None or not new.strip():
            final[key] = original
            continue
        new = new.strip()
        problems = find_violations(resume.raw_text, new, allowed) if new != original else []
        if problems:
            final[key] = original
            reverted.add(key)
            changelog.append(
                ChangeEntry(
                    section=key,
                    action="reverted",
                    detail="Откат к оригиналу: " + "; ".join(problems),
                )
            )
            warnings.append(
                f"Раздел «{key}»: LLM добавила неподтверждённые факты — {'; '.join(problems)}"
            )
        else:
            final[key] = new
    for key in out.sections:
        if key not in resume.sections:
            warnings.append(f"LLM добавила новый раздел «{key}» — проигнорирован")

    for ch in out.changes:
        if ch.section in reverted or ch.section not in final:
            continue
        if final[ch.section] == resume.sections.get(ch.section) and ch.action != "unchanged":
            continue
        changelog.append(ch)
    described = {c.section for c in changelog}
    for key in final:
        if key not in described and final[key] != resume.sections.get(key):
            changelog.append(
                ChangeEntry(
                    section=key, action="rephrased", detail="Текст изменён (без описания от модели)"
                )
            )

    exp_orig = resume.section("experience")
    if exp_orig and len(final.get("experience", "")) < 0.5 * len(exp_orig):
        warnings.append(
            "Раздел опыта сокращён более чем вдвое — проверьте, не потерялись ли места работы."
        )

    final = _ordered_sections(final, platform)
    before = score_resume(resume, vacancy).total
    after = score_resume(resume_from_sections(final, resume.language), vacancy).total
    return TailoredResume(
        platform=platform,
        vacancy_title=vacancy.title,
        language=resume.language,
        sections=final,
        changelog=changelog,
        warnings=warnings,
        score_before=before,
        score_after=after,
    )


# ------------------------------------------------------------------ cover letter


def cover_letter(
    resume: Resume, vacancy: Vacancy, llm: LLM, platform: str = "generic"
) -> CoverLetter:
    if platform not in COVER_RULES:
        raise ValueError(f"Неизвестная платформа: {platform}")
    present = present_keywords(resume, vacancy)
    payload = {
        "task": "cover_letter",
        "platform": platform,
        "platform_rules": COVER_RULES[platform],
        "language": vacancy.language,
        "candidate_name": resume.contacts.name,
        "vacancy": {"title": vacancy.title, "must_have": vacancy.must_have},
        "present_keywords": present,
        "missing_keywords": [k for k in vacancy.keywords if k not in present],
        "highlights": _highlights(resume),
        "resume": {k: v for k, v in resume.sections.items() if k in SECTION_NAMES},
    }
    prompt = (
        f"Напиши сопроводительное письмо на позицию «{vacancy.title}» ({platform}).\n\n"
        + embed_input(payload)
    )
    raw = llm.complete(prompt, system=COVER_SYSTEM)
    text = raw if isinstance(raw, str) else str(raw.get("text", ""))
    text = text.strip()

    # Письмо может упоминать компанию и формулировки вакансии, но не новые навыки/опыт.
    warnings: list[str] = []
    kept: list[str] = []
    for line in text.splitlines():
        problems = find_violations(resume.raw_text, line, vacancy.raw_text + "\n" + vacancy.title)
        skill_problems = [p for p in problems if p.startswith(("новые навыки", "новые числа"))]
        if skill_problems:
            warnings.append(
                f"Удалена строка с неподтверждёнными фактами: {line[:80]!r} ({'; '.join(skill_problems)})"
            )
            continue
        kept.append(line)
    return CoverLetter(
        platform=platform,
        vacancy_title=vacancy.title,
        text="\n".join(kept).strip(),
        used_keywords=[k for k in present if contains_keyword(text, k)],
        warnings=warnings,
    )
