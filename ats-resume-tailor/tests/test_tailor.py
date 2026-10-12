from __future__ import annotations

import pytest

from ats_tailor.llm import MockLLM, extract_input
from ats_tailor.tailor import (
    cover_letter,
    extract_organizations,
    find_violations,
    tailor_resume,
)


def test_mock_tailor_changelog_and_truthfulness(resume_ru, vacancy_ru) -> None:
    llm = MockLLM()
    tr = tailor_resume(resume_ru, vacancy_ru, llm, platform="hh")

    assert tr.platform == "hh"
    assert set(tr.sections) == set(resume_ru.sections)
    actions = {(c.section, c.action) for c in tr.changelog}
    assert ("skills", "reordered") in actions
    assert ("summary", "rephrased") in actions
    assert tr.warnings == []

    # опыт и образование — дословно, компании не придуманы
    assert tr.sections["experience"] == resume_ru.section("experience")
    assert tr.sections["education"] == resume_ru.section("education")
    full_after = "\n".join(tr.sections.values())
    orgs_before = {o.lower() for o in extract_organizations(resume_ru.raw_text)}
    orgs_after = {o.lower() for o in extract_organizations(full_after)}
    title_words = vacancy_ru.title.lower()
    assert all(o in orgs_before or o in title_words for o in orgs_after)

    # отсутствующие в резюме навыки не добавлены
    for missing in ("Kubernetes", "MLflow", "LlamaIndex", "vLLM"):
        assert missing not in tr.sections["skills"]
    # релевантное — в начале навыков
    assert tr.sections["skills"].split(",")[0].strip() in vacancy_ru.keywords
    assert tr.score_after is not None and tr.score_before is not None
    assert tr.score_after >= tr.score_before

    # LLM получила жёсткие правила и списки ключевых слов
    payload = extract_input(llm.calls[0]["prompt"])
    assert payload is not None
    assert "Kubernetes" in payload["missing_keywords"]
    assert "Python" in payload["present_keywords"]
    assert "Никогда не выдумывай" in llm.calls[0]["system"]


def test_invented_company_is_reverted(resume_ru, vacancy_ru) -> None:
    fake_exp = resume_ru.section("experience") + (
        "\n\n### ООО «Яндекс» — Lead ML Engineer\n2015 — 2019\n- Руководил командой 20 человек."
    )
    llm = MockLLM(
        responses=[
            {
                "sections": {
                    "experience": fake_exp,
                    "skills": "Python, Kubernetes, FastAPI",
                },
                "changes": [
                    {"section": "experience", "action": "rephrased", "detail": "улучшил"},
                    {"section": "skills", "action": "keywords_added", "detail": "Kubernetes"},
                ],
            }
        ]
    )
    tr = tailor_resume(resume_ru, vacancy_ru, llm, platform="generic")
    assert tr.sections["experience"] == resume_ru.section("experience")
    assert tr.sections["skills"] == resume_ru.section("skills")
    reverted = {c.section for c in tr.changelog if c.action == "reverted"}
    assert reverted == {"experience", "skills"}
    assert not any(c.action == "rephrased" and c.section == "experience" for c in tr.changelog)
    assert any("Яндекс" in w for w in tr.warnings)
    assert any("Kubernetes" in w for w in tr.warnings)


def test_new_section_and_garbage(resume_ru, vacancy_ru) -> None:
    tr = tailor_resume(
        resume_ru,
        vacancy_ru,
        MockLLM(responses=[{"sections": {"awards": "Нобелевская премия"}, "changes": []}]),
    )
    assert "awards" not in tr.sections
    assert any("awards" in w for w in tr.warnings)

    tr2 = tailor_resume(resume_ru, vacancy_ru, MockLLM(responses=["полная ерунда"]))
    assert tr2.sections == {k: resume_ru.sections[k] for k in tr2.sections}  # ничего не изменено
    assert tr2.warnings


def test_platform_order_and_validation(resume_ru, vacancy_ru) -> None:
    tr = tailor_resume(resume_ru, vacancy_ru, MockLLM(), platform="linkedin")
    assert next(iter(tr.sections)) == "summary"
    with pytest.raises(ValueError):
        tailor_resume(resume_ru, vacancy_ru, MockLLM(), platform="myspace")


def test_en_resume_tailor(resume_en, vacancy_en) -> None:
    tr = tailor_resume(resume_en, vacancy_en, MockLLM(), platform="indeed")
    assert tr.language == "en"
    assert "Relevant for the" in tr.sections["summary"]
    assert tr.sections["experience"] == resume_en.section("experience")
    assert tr.warnings == []


def test_find_violations_unit() -> None:
    original = "ООО «Ромашка», Python, Docker, 2020 — 2022, ускорил на 30%"
    assert find_violations(original, "Python и Docker в ООО «Ромашка», 30%") == []
    problems = find_violations(original, "Kubernetes в ООО «Рога и копыта», 45%")
    joined = " ".join(problems)
    assert "Kubernetes" in joined
    assert "45%" in joined
    assert "Рога и копыта" in joined
    # английский текст: «at Google» — новая компания
    assert find_violations("Worked at Acme on Python", "Worked at Google on Python")


def test_cover_letter_mock(resume_ru, vacancy_ru) -> None:
    cl = cover_letter(resume_ru, vacancy_ru, MockLLM(), platform="hh")
    assert vacancy_ru.title in cl.text
    assert "Алексей Смирнов" in cl.text
    assert "Python" in cl.used_keywords
    assert "Kubernetes" not in cl.text
    assert cl.warnings == []


def test_cover_letter_en(resume_en, vacancy_en) -> None:
    cl = cover_letter(resume_en, vacancy_en, MockLLM(), platform="linkedin")
    assert cl.text.startswith("Dear Hiring Team")
    assert "Maria Ivanova" in cl.text


def test_cover_letter_strips_invented_claims(resume_ru, vacancy_ru) -> None:
    letter = (
        "Здравствуйте!\n"
        "У меня 9 лет опыта с Kubernetes.\n"
        "Работаю с Python и FastAPI.\n"
        "С уважением, Алексей"
    )
    cl = cover_letter(resume_ru, vacancy_ru, MockLLM(responses=[letter]))
    assert "Kubernetes" not in cl.text
    assert "Работаю с Python и FastAPI." in cl.text
    assert cl.warnings
