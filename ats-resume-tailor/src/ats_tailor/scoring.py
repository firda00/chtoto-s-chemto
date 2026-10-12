"""Расчёт ATS Score (0–100) с разбивкой по компонентам.

total = Σ weight_i * score_i, где score_i ∈ [0, 1], Σ weight_i = 100:

* keywords   (40) — покрытие ключевых слов: 0.75 * must_have + 0.25 * nice_to_have (с синонимами)
* similarity (20) — TF-IDF cosine (char 3–5-граммы) резюме и вакансии: min(1, cos / 0.4)
* sections   (15) — полнота разделов: опыт .30, навыки .25, контакты .20, образование .15, о себе .10
* experience (15) — годы опыта: min(1, years_resume / years_required); 0.5 если даты не найдены
* formatting (10) — 1.0 минус штрафы за таблицы, картинки, длину, отсутствие контактов
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import ATSScore, Resume, ScoreComponent, Vacancy
from .skills import contains_keyword, normalize_for_matching

WEIGHTS: dict[str, float] = {
    "keywords": 40.0,
    "similarity": 20.0,
    "sections": 15.0,
    "experience": 15.0,
    "formatting": 10.0,
}
MUST_SHARE = 0.75
SIM_FULL = 0.4  # cosine >= 0.4 считается полным совпадением лексики
SECTION_WEIGHTS: dict[str, float] = {
    "experience": 0.30,
    "skills": 0.25,
    "contacts": 0.20,
    "education": 0.15,
    "summary": 0.10,
}
SECTION_TITLES_RU = {
    "experience": "Опыт работы",
    "skills": "Навыки",
    "contacts": "Контакты",
    "education": "Образование",
    "summary": "О себе",
}

# Штрафы форматирования (вычитаются из 1.0)
PENALTIES: dict[str, float] = {
    "tables": 0.30,
    "images": 0.15,
    "too_long": 0.20,
    "too_short": 0.20,
    "too_many_pages": 0.10,
    "no_email": 0.20,
    "no_phone": 0.10,
    "no_headings": 0.15,
    "no_text": 1.00,
}
MAX_WORDS = 1100
MIN_WORDS = 150

STOPWORDS = {
    # ru
    "и",
    "в",
    "во",
    "на",
    "с",
    "со",
    "по",
    "для",
    "от",
    "до",
    "из",
    "к",
    "о",
    "об",
    "а",
    "но",
    "или",
    "не",
    "что",
    "как",
    "это",
    "мы",
    "вы",
    "ты",
    "он",
    "она",
    "они",
    "я",
    "у",
    "за",
    "при",
    "без",
    "над",
    "под",
    "же",
    "ли",
    "бы",
    "то",
    "так",
    "также",
    "уже",
    "все",
    "всё",
    "вас",
    "нас",
    "наш",
    "наша",
    "наши",
    "ваш",
    "его",
    "ее",
    "её",
    "их",
    "который",
    "которые",
    "будет",
    "быть",
    "есть",
    "лет",
    "года",
    "год",
    "более",
    "менее",
    "очень",
    "свой",
    "свои",
    # en
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "by",
    "at",
    "from",
    "is",
    "are",
    "be",
    "as",
    "we",
    "you",
    "our",
    "your",
    "will",
    "that",
    "this",
    "it",
    "its",
    "have",
    "has",
    "can",
    "who",
    "what",
    "about",
    "into",
    "than",
    "more",
    "years",
    "year",
}


# ------------------------------------------------------------------ experience years

_MONTHS = {
    "янв": 1,
    "фев": 2,
    "мар": 3,
    "апр": 4,
    "май": 5,
    "мая": 5,
    "июн": 6,
    "июл": 7,
    "авг": 8,
    "сен": 9,
    "окт": 10,
    "ноя": 11,
    "дек": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_MONTH_WORD = r"(?:[A-Za-zА-Яа-яЁё]{3,9}\.?)"


def _date_re(p: str) -> str:
    return rf"(?:(?P<{p}m>{_MONTH_WORD})\s+|(?P<{p}n>\d{{1,2}})[./])?(?P<{p}y>(?:19|20)\d{{2}})"


_PRESENT = (
    r"(?P<present>по\s+настоящее\s+время|настоящее\s+время|наст\.?\s*вр(?:емя)?\.?|н\.\s?в\.|"
    r"сейчас|present|current|now|today)"
)
_RANGE = re.compile(
    _date_re("s") + r"\s*(?:-|–|—|по|to|until)\s*(?:" + _date_re("e") + "|" + _PRESENT + ")",
    re.I,
)
_YEARS_PHRASE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*\+?\s*(?:лет|года|год|years?)\s+(?:опыта|коммерческого|of\s+experience|experience)",
    re.I,
)


def _month(word: str | None, num: str | None, default: int) -> int:
    if num:
        n = int(num)
        return n if 1 <= n <= 12 else default
    if word:
        return _MONTHS.get(word.lower().rstrip(".")[:3], default)
    return default


def _merge_months(intervals: list[tuple[int, int]]) -> int:
    total = 0
    cur_s: int | None = None
    cur_e = 0
    for s, e in sorted(intervals):
        if cur_s is None or s > cur_e:
            if cur_s is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_s is not None:
        total += cur_e - cur_s
    return total


def estimate_years(text: str, today: date | None = None) -> float | None:
    """Оценить суммарный опыт по диапазонам дат (пересечения склеиваются)."""
    today = today or date.today()
    now_idx = today.year * 12 + today.month
    intervals: list[tuple[int, int]] = []
    for m in _RANGE.finditer(text):
        sy = int(m.group("sy"))
        sm = _month(m.group("sm"), m.group("sn"), 1)
        start = sy * 12 + sm
        if m.group("present"):
            end = now_idx
        else:
            end = int(m.group("ey")) * 12 + _month(m.group("em"), m.group("en"), 12)
        end = min(end, now_idx)
        if end >= start:
            intervals.append((start, end + 1))  # месяцы считаются включительно
    if intervals:
        return round(_merge_months(intervals) / 12, 1)
    m = _YEARS_PHRASE.search(text)
    if m:
        return float(m.group(1).replace(",", "."))
    return None


# ------------------------------------------------------------------ components


@dataclass
class KeywordResult:
    score: float
    matched: list[str]
    missing_must: list[str]
    missing_nice: list[str]
    details: str


def _units(items: list[str], groups: list[list[str]]) -> list[list[str]]:
    """Свернуть взаимозаменяемые навыки в одно требование: [[LangChain, LlamaIndex], [Docker]]."""
    units: list[list[str]] = []
    used: set[str] = set()
    for item in items:
        if item in used:
            continue
        group = next((g for g in groups if item in g and all(x in items for x in g)), None)
        unit = group or [item]
        used.update(unit)
        units.append(unit)
    return units


def keyword_coverage(resume_text: str, vacancy: Vacancy) -> KeywordResult:
    must = list(vacancy.must_have)
    nice = list(vacancy.nice_to_have)
    if not must and not nice:
        must = list(vacancy.keywords)
    must_units = _units(must, vacancy.alternatives)
    nice_units = _units(nice, vacancy.alternatives)

    def split(units: list[list[str]]) -> tuple[list[str], list[str], int]:
        matched: list[str] = []
        missing: list[str] = []
        ok = 0
        for unit in units:
            hits = [k for k in unit if contains_keyword(resume_text, k)]
            if hits:
                ok += 1
                matched.extend(hits)
            else:
                missing.append(" / ".join(unit))
        return matched, missing, ok

    found_must, missing_must, ok_must = split(must_units)
    found_nice, missing_nice, ok_nice = split(nice_units)

    must_cov = ok_must / len(must_units) if must_units else None
    nice_cov = ok_nice / len(nice_units) if nice_units else None
    if must_cov is None and nice_cov is None:
        score = 1.0
    elif nice_cov is None:
        score = must_cov  # type: ignore[assignment]
    elif must_cov is None:
        score = nice_cov
    else:
        score = MUST_SHARE * must_cov + (1 - MUST_SHARE) * nice_cov
    details = f"must-have: {ok_must}/{len(must_units)}, nice-to-have: {ok_nice}/{len(nice_units)}"
    return KeywordResult(
        score=float(score),
        matched=found_must + found_nice,
        missing_must=missing_must,
        missing_nice=missing_nice,
        details=details,
    )


def tfidf_similarity(resume_text: str, vacancy_text: str) -> float:
    """Косинусная близость TF-IDF (символьные 3–5-граммы) после нормализации синонимов."""
    docs = [normalize_for_matching(resume_text), normalize_for_matching(vacancy_text)]
    if not all(d.strip() for d in docs):
        return 0.0
    # Символьные n-граммы внутри слов устойчивы к русской морфологии
    # («тестов»/«тесты», «модели»/«моделями») и к смешению RU/EN.
    vec = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    try:
        matrix = vec.fit_transform(docs)
    except ValueError:  # пустой словарь
        return 0.0
    return float(cosine_similarity(matrix[0], matrix[1])[0, 0])


def top_missing_terms(resume_text: str, vacancy_text: str, n: int = 8) -> list[str]:
    """Самые «весомые» слова вакансии, которых нет в резюме (подсказка для перефразирования)."""
    vec = TfidfVectorizer(
        lowercase=True, token_pattern=r"(?u)[^\W\d_][\w+#]{3,}", stop_words=sorted(STOPWORDS)
    )
    try:
        matrix = vec.fit_transform([vacancy_text, resume_text])
    except ValueError:
        return []
    vocab = vec.get_feature_names_out()
    res_low = resume_text.lower()
    weights = matrix[0].toarray()[0]
    ranked = sorted(range(len(vocab)), key=lambda i: -weights[i])
    out = [vocab[i] for i in ranked if weights[i] > 0 and vocab[i] not in res_low]
    return out[:n]


def section_completeness(resume: Resume) -> tuple[float, list[str]]:
    missing = [s for s in SECTION_WEIGHTS if not resume.section(s).strip()]
    score = sum(w for s, w in SECTION_WEIGHTS.items() if s not in missing)
    return round(score, 4), missing


def formatting_flags(resume: Resume) -> tuple[float, list[str]]:
    flags: list[str] = []
    text = resume.raw_text
    words = len(re.findall(r"\w+", text))
    doc = resume.doc
    if words < 20:
        flags.append("no_text")
    if doc is not None:
        if doc.tables:
            flags.append("tables")
        if doc.images:
            flags.append("images")
        if doc.pages and doc.pages > 2:
            flags.append("too_many_pages")
    if words > MAX_WORDS:
        flags.append("too_long")
    elif 20 <= words < MIN_WORDS:
        flags.append("too_short")
    if not resume.contacts.emails:
        flags.append("no_email")
    if not resume.contacts.phones:
        flags.append("no_phone")
    if "other" in resume.sections and len(resume.sections) == 1:
        flags.append("no_headings")
    score = max(0.0, 1.0 - sum(PENALTIES[f] for f in flags))
    return round(score, 4), flags


RED_FLAG_TEXT = {
    "tables": "Таблицы: многие ATS читают их построчно и перемешивают текст",
    "images": "Изображения/иконки: ATS их не читает, текст внутри картинок теряется",
    "too_long": f"Резюме длиннее {MAX_WORDS} слов (≈ больше 2 страниц)",
    "too_short": f"Резюме короче {MIN_WORDS} слов — мало контекста для ATS",
    "too_many_pages": "Больше 2 страниц PDF",
    "no_email": "Не найден email",
    "no_phone": "Не найден телефон",
    "no_headings": "Не распознаны стандартные заголовки разделов",
    "no_text": "Текст почти не извлекается (скан или картинка?)",
}

FLAG_FIX = {
    "tables": "Замените таблицы на обычные абзацы и маркированные списки.",
    "images": "Уберите иконки/фото/графики, продублируйте важное текстом.",
    "too_long": "Сократите до 1–2 страниц: оставьте последние 10 лет и релевантные проекты.",
    "too_short": "Раскройте опыт: 3–5 пунктов с результатами на каждое место работы.",
    "too_many_pages": "Сожмите документ до 2 страниц.",
    "no_email": "Добавьте email в шапку резюме.",
    "no_phone": "Добавьте телефон в международном формате (+7 …, +998 …).",
    "no_headings": "Используйте стандартные заголовки: «Опыт работы», «Навыки», «Образование».",
    "no_text": "Сохраните резюме как текстовый PDF/DOCX, а не скан.",
}


def _component(name: str, score: float, details: str) -> ScoreComponent:
    score = max(0.0, min(1.0, score))
    weight = WEIGHTS[name]
    return ScoreComponent(
        name=name,
        weight=weight,
        score=round(score, 4),
        points=round(weight * score, 2),
        details=details,
    )


def score_resume(resume: Resume, vacancy: Vacancy, today: date | None = None) -> ATSScore:
    text = resume.raw_text
    vac_text = vacancy.raw_text or " ".join([vacancy.title, *vacancy.keywords])

    kw = keyword_coverage(text, vacancy)

    cos = tfidf_similarity(text, vac_text)
    sim_score = min(1.0, cos / SIM_FULL)

    sec_score, missing_sections = section_completeness(resume)
    fmt_score, flags = formatting_flags(resume)

    exp_text = resume.section("experience") or text
    years = estimate_years(exp_text, today=today)
    if years is None and resume.section("summary"):
        years = estimate_years(resume.section("summary"), today=today)
    req = vacancy.min_years
    if not req:
        exp_score, exp_details = 1.0, "требование к опыту не указано"
    elif years is None:
        exp_score, exp_details = 0.5, f"требуется {req:g} лет, даты в резюме не распознаны"
    else:
        exp_score = min(1.0, years / req)
        exp_details = f"в резюме ≈ {years:g} лет, требуется {req:g}"

    components = [
        _component("keywords", kw.score, kw.details),
        _component("similarity", sim_score, f"cosine TF-IDF = {cos:.3f}"),
        _component(
            "sections",
            sec_score,
            "все разделы на месте"
            if not missing_sections
            else "нет: " + ", ".join(SECTION_TITLES_RU[s] for s in missing_sections),
        ),
        _component("experience", exp_score, exp_details),
        _component("formatting", fmt_score, "без замечаний" if not flags else ", ".join(flags)),
    ]
    total = round(sum(c.points for c in components), 1)

    suggestions = build_suggestions(
        kw, cos, missing_sections, flags, years, req, top_missing_terms(text, vac_text)
    )
    return ATSScore(
        total=max(0.0, min(100.0, total)),
        components=components,
        matched_keywords=kw.matched,
        missing_must_have=kw.missing_must,
        missing_nice_to_have=kw.missing_nice,
        red_flags=[RED_FLAG_TEXT[f] for f in flags],
        resume_years=years,
        required_years=req,
        suggestions=suggestions,
    )


def build_suggestions(
    kw: KeywordResult,
    cos: float,
    missing_sections: list[str],
    flags: list[str],
    years: float | None,
    req: float | None,
    missing_terms: list[str],
) -> list[str]:
    out: list[str] = []
    for k in kw.missing_must:
        out.append(
            f"Обязательный навык «{k}» не найден. Если он у вас есть — добавьте его в «Навыки» "
            f"и подкрепите пунктом в опыте (что сделали с {k} и какой результат)."
        )
    if kw.missing_nice:
        out.append(
            "Желательные навыки, которых нет в резюме: "
            + ", ".join(kw.missing_nice)
            + ". Упомяните те, с которыми реально работали (хотя бы в pet-проектах)."
        )
    if req and years is not None and years < req:
        out.append(
            f"Вакансия требует от {req:g} лет опыта, в резюме найдено ≈ {years:g}. "
            "Проверьте, что у всех мест работы указаны даты (мм.гггг — мм.гггг), "
            "и добавьте релевантные проекты/фриланс, если они были."
        )
    elif req and years is None:
        out.append(
            "Не удалось распознать даты работы: укажите периоды в формате «03.2021 — 06.2023»."
        )
    for s in missing_sections:
        out.append(f"Добавьте раздел «{SECTION_TITLES_RU[s]}» со стандартным заголовком.")
    for f in flags:
        out.append(FLAG_FIX[f])
    if cos < SIM_FULL * 0.6 and missing_terms:
        out.append(
            "Лексика резюме слабо пересекается с вакансией. Используйте формулировки вакансии "
            "там, где это правда, например: " + ", ".join(missing_terms[:6]) + "."
        )
    return out


def score_texts(resume_text: str, vacancy_text: str, today: date | None = None) -> ATSScore:
    from .parsers import parse_resume
    from .vacancy import extract_vacancy_rules

    return score_resume(parse_resume(resume_text), extract_vacancy_rules(vacancy_text), today=today)
