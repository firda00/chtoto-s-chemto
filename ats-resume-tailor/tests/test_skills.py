from __future__ import annotations

import pytest

from ats_tailor.skills import (
    canonicalize,
    contains_keyword,
    find_skills,
    find_tech_tokens,
    normalize_for_matching,
)


@pytest.mark.parametrize(
    ("variant", "canon"),
    [
        ("python3", "Python"),
        ("Large Language Model", "LLM"),
        ("k8s", "Kubernetes"),
        ("postgres", "PostgreSQL"),
        ("sklearn", "scikit-learn"),
        ("retrieval-augmented generation", "RAG"),
        ("Unknown Tool", "Unknown Tool"),
    ],
)
def test_canonicalize(variant: str, canon: str) -> None:
    assert canonicalize(variant) == canon


@pytest.mark.parametrize(
    ("text", "keyword", "expected"),
    [
        ("Опыт с python3 и asyncio", "Python", True),
        ("Built apps on large language models", "LLM", True),
        ("Работа с большими языковыми моделями", "LLM", True),
        ("Deploying to k8s clusters", "Kubernetes", True),
        ("PostgreSQL, Redis", "SQL", False),  # SQL не находится внутри PostgreSQL
        ("JavaScript, TypeScript", "Java", False),
        ("Python-разработчик", "Python", True),
        ("HTML markup", "Machine Learning", False),  # «ml» в html — не машинное обучение
        ("5 лет в ML", "Machine Learning", True),
        ("C++ и C#", "C++", True),
        ("RAG-система поиска", "RAG", True),
        ("проводил A/B-тесты", "A/B testing", True),
        ("ClickHouse, Kafka", "ClickHouse", True),  # вне словаря — точное совпадение
    ],
)
def test_contains_keyword(text: str, keyword: str, expected: bool) -> None:
    assert contains_keyword(text, keyword) is expected


def test_find_skills() -> None:
    found = find_skills("Python3, FastAPI, docker-compose, pgvector, LLM-агенты")
    assert {"Python", "FastAPI", "Docker", "Vector DB", "AI Agents", "LLM"} <= found


def test_find_tech_tokens() -> None:
    toks = find_tech_tokens("Experience with ClickHouse, ONNX and TensorRT. The team uses Python.")
    assert {"ClickHouse", "ONNX", "TensorRT"} <= toks
    assert "Python" not in toks  # уже в словаре
    assert "The" not in toks


def test_normalize_for_matching() -> None:
    norm = normalize_for_matching("python3 and Large Language Models").lower()
    assert "python" in norm
    assert "llm" in norm
