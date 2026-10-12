from __future__ import annotations

from datetime import date

import pytest

from ats_tailor.models import DocumentInfo, Vacancy
from ats_tailor.parsers import parse_resume
from ats_tailor.scoring import (
    PENALTIES,
    WEIGHTS,
    estimate_years,
    formatting_flags,
    keyword_coverage,
    score_resume,
    score_texts,
    section_completeness,
    tfidf_similarity,
)

TODAY = date(2026, 10, 1)

FULL_RESUME = """Иван Петров
ivan@example.com | +7 900 111-22-33

## О себе
Python-разработчик.

## Навыки
Python, Docker, MLflow

## Опыт работы
ООО Ромашка — разработчик
03.2020 — 02.2022
- Писал сервисы на Python.

## Образование
МГУ, 2015 — 2019
"""


def test_weights_sum_to_100() -> None:
    assert sum(WEIGHTS.values()) == 100


def test_sample_ru_score(resume_ru, vacancy_ru) -> None:
    s = score_resume(resume_ru, vacancy_ru, today=TODAY)
    assert 70 <= s.total <= 95
    assert {c.name for c in s.components} == set(WEIGHTS)
    assert s.total == pytest.approx(sum(c.points for c in s.components), abs=0.11)
    assert s.missing_must_have == []
    assert {"Kubernetes", "MLflow", "CI/CD"} <= set(s.missing_nice_to_have)
    assert "Python" in s.matched_keywords
    assert s.component("sections").score == 1.0
    assert s.component("experience").score == 1.0
    assert s.resume_years is not None and s.resume_years >= 5


def test_sample_en_score(resume_en, vacancy_en) -> None:
    s = score_resume(resume_en, vacancy_en, today=TODAY)
    assert 75 <= s.total <= 100
    # в EN-резюме нет слова «embeddings» — это единственный пропущенный must-have
    assert s.missing_must_have == ["Embeddings"]
    assert "LangChain / LlamaIndex" in s.missing_nice_to_have


def test_keyword_coverage_known_inputs() -> None:
    v = Vacancy(
        must_have=["Python", "Docker", "Kubernetes", "SQL"],
        nice_to_have=["MLflow", "Airflow"],
    )
    kw = keyword_coverage("Python, Docker, MLflow", v)
    # 0.75 * 2/4 + 0.25 * 1/2
    assert kw.score == pytest.approx(0.5)
    assert kw.missing_must == ["Kubernetes", "SQL"]
    assert kw.missing_nice == ["Airflow"]
    assert kw.matched == ["Python", "Docker", "MLflow"]


def test_keyword_coverage_synonyms() -> None:
    v = Vacancy(must_have=["Python", "LLM", "Kubernetes", "PostgreSQL", "RAG"])
    text = "python3, large language models, k8s, Postgres, retrieval-augmented generation"
    kw = keyword_coverage(text, v)
    assert kw.score == 1.0
    assert kw.missing_must == []


def test_keyword_coverage_alternatives() -> None:
    v = Vacancy(
        must_have=["LangChain", "LlamaIndex", "Docker"],
        alternatives=[["LangChain", "LlamaIndex"]],
    )
    assert keyword_coverage("LangChain, Docker", v).score == 1.0
    kw = keyword_coverage("Docker", v)
    assert kw.score == pytest.approx(0.5)
    assert kw.missing_must == ["LangChain / LlamaIndex"]


def test_keyword_coverage_without_requirements() -> None:
    assert keyword_coverage("что угодно", Vacancy()).score == 1.0


def test_estimate_years_ranges_and_overlap() -> None:
    assert estimate_years("03.2020 — 02.2022", today=TODAY) == 2.0
    # пересекающиеся периоды не суммируются дважды: 2019-01..2021-12 = 3 года
    assert estimate_years("2019 — 2020\n01.2020 — 12.2021", today=TODAY) == 3.0
    assert estimate_years("Jan 2020 - Present", today=date(2022, 1, 1)) == pytest.approx(2.1)
    assert estimate_years("март 2021 — по настоящее время", today=date(2022, 3, 1)) == 1.1
    assert estimate_years("5+ years of experience in ML", today=TODAY) == 5
    assert estimate_years("без дат", today=TODAY) is None


def test_experience_component_partial() -> None:
    v = Vacancy(must_have=["Python"], min_years=5)
    s = score_resume(parse_resume(FULL_RESUME), v, today=TODAY)
    exp = s.component("experience")
    assert exp.score == pytest.approx(2.0 / 5)
    assert s.resume_years == 2.0
    assert any("от 5 лет" in sug for sug in s.suggestions)


def test_section_completeness() -> None:
    r = parse_resume(FULL_RESUME.split("## Образование")[0])
    score, missing = section_completeness(r)
    assert missing == ["education"]
    assert score == pytest.approx(0.85)


def test_formatting_flags() -> None:
    r = parse_resume("Навыки\nPython " * 30)
    r.doc = DocumentInfo(text=r.raw_text, fmt="docx", tables=1, images=2)
    score, flags = formatting_flags(r)
    assert {"tables", "images", "no_email", "no_phone"} <= set(flags)
    expected = 1 - sum(PENALTIES[f] for f in flags)
    assert score == pytest.approx(max(0.0, expected))


def test_formatting_too_long() -> None:
    text = FULL_RESUME + ("слово " * 1200)
    _, flags = formatting_flags(parse_resume(text))
    assert "too_long" in flags
    assert "no_email" not in flags


def test_similarity_properties(resume_ru, vacancy_ru) -> None:
    same = tfidf_similarity(resume_ru.raw_text, resume_ru.raw_text)
    assert same == pytest.approx(1.0)
    related = tfidf_similarity(resume_ru.raw_text, vacancy_ru.raw_text)
    unrelated = tfidf_similarity(
        "Повар-кондитер, выпечка тортов, работа с тестом", vacancy_ru.raw_text
    )
    assert related > unrelated
    assert tfidf_similarity("", "текст") == 0.0


def test_unrelated_resume_scores_low(vacancy_ru) -> None:
    cook = parse_resume(
        "Мария\n## О себе\nПовар-кондитер, 2 года опыта.\n## Опыт работы\nКафе «Ромашка», 2022 — 2024"
    )
    s = score_resume(cook, vacancy_ru, today=TODAY)
    assert s.total < 40
    assert "Python" in s.missing_must_have
    assert any("«Python»" in sug for sug in s.suggestions)
    assert any("email" in sug.lower() for sug in s.suggestions)


def test_score_texts_shortcut(resume_ru_path, vacancy_ru_path) -> None:
    s = score_texts(
        resume_ru_path.read_text(encoding="utf-8"),
        vacancy_ru_path.read_text(encoding="utf-8"),
        today=TODAY,
    )
    assert 0 <= s.total <= 100
    assert s.missing_keywords == s.missing_must_have + s.missing_nice_to_have
