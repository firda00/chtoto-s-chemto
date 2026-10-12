from __future__ import annotations

from pathlib import Path

import pytest

from ats_tailor.llm import MockLLM
from ats_tailor.vacancy import extract_min_years, extract_vacancy, extract_vacancy_rules


def test_ru_vacancy_rules(vacancy_ru) -> None:
    v = vacancy_ru
    assert v.title == "LLM Engineer (Middle+/Senior)"
    assert {"Python", "LLM", "RAG", "FastAPI", "Docker", "PostgreSQL", "Vector DB"} <= set(
        v.must_have
    )
    assert {"Kubernetes", "CI/CD", "MLflow", "vLLM", "Ollama", "Fine-tuning"} <= set(v.nice_to_have)
    assert not set(v.must_have) & set(v.nice_to_have)
    assert v.min_years == 3
    assert ["LangChain", "LlamaIndex"] in v.alternatives
    assert v.language == "ru"
    # «AI Agents» встречается только в обязанностях -> ключевое слово, но не must
    assert "AI Agents" in v.keywords
    assert "AI Agents" not in v.must_have


def test_en_vacancy_rules(vacancy_en) -> None:
    v = vacancy_en
    assert v.title == "Senior AI Engineer (LLM Applications)"
    assert v.min_years == 4
    assert {"Python", "SQL", "LLM", "RAG", "Embeddings", "Kubernetes", "AWS", "GCP"} <= set(
        v.must_have
    )
    assert {"LangChain", "MLflow", "Airflow", "vLLM", "Fine-tuning"} <= set(v.nice_to_have)
    assert ["AWS", "GCP"] in v.alternatives
    assert v.language == "en"


def test_ru_ml_vacancy_tech_tokens(data_dir: Path) -> None:
    v = extract_vacancy_rules((data_dir / "vacancies" / "ml_engineer_ru.txt").read_text("utf-8"))
    assert v.min_years == 5
    assert "ClickHouse" in v.nice_to_have  # нет в словаре — найден как тех-токен
    assert ["PyTorch", "TensorFlow"] in v.alternatives
    assert v.title.startswith("Senior ML Engineer")


@pytest.mark.parametrize(
    ("line", "years"),
    [
        ("Опыт коммерческой разработки от 3 лет", 3),
        ("3+ years of experience with Python", 3),
        ("Experience: 2-4 years in backend", 2),
        ("Опыт работы не менее 1,5 года", 1.5),
        ("Стек: Python 3.12, Docker", None),
    ],
)
def test_extract_min_years(line: str, years: float | None) -> None:
    assert extract_min_years([line]) == years


def test_unstructured_vacancy_inline_plus() -> None:
    text = (
        "Ищем Python-разработчика с опытом от 2 лет.\n"
        "Нужно знать Django и PostgreSQL.\n"
        "Знание Kubernetes будет плюсом."
    )
    v = extract_vacancy_rules(text)
    assert {"Python", "Django", "PostgreSQL"} <= set(v.must_have)
    assert v.nice_to_have == ["Kubernetes"]
    assert v.min_years == 2


def test_llm_extraction_is_merged_and_grounded(vacancy_ru_path: Path) -> None:
    text = vacancy_ru_path.read_text(encoding="utf-8")
    llm = MockLLM(
        responses=[
            {
                "title": "LLM Engineer",
                "must_have": ["python3", "Rust", "large language model"],
                "nice_to_have": ["Kubernetes", "Haskell"],
                "min_years": 3,
                "keywords": ["Rust"],
            }
        ]
    )
    v = extract_vacancy(text, llm)
    assert v.title == "LLM Engineer"
    assert "Rust" not in v.must_have + v.keywords  # галлюцинация LLM отброшена
    assert "Haskell" not in v.nice_to_have
    assert "Python" in v.must_have and "LLM" in v.must_have
    assert "FastAPI" in v.must_have  # правила сохраняются
    # схема отдана LLM как JSON Schema
    assert llm.calls[0]["json_schema"]["properties"]["must_have"]["type"] == "array"


def test_llm_garbage_falls_back_to_rules(vacancy_ru_path: Path) -> None:
    text = vacancy_ru_path.read_text(encoding="utf-8")
    v = extract_vacancy(text, MockLLM(responses=["это не JSON"]))
    assert v == extract_vacancy_rules(text)
