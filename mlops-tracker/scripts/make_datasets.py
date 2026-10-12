"""Генерирует демо-датасеты в data/evals/ (синтетические, RU/EN).

uv run python scripts/make_datasets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "evals"

QA = [
    {
        "id": "qa-capital-fr",
        "input": "What is the capital of France?",
        "expected": "Paris",
        "metadata": {"lang": "en", "topic": "geo"},
    },
    {
        "id": "qa-capital-uz",
        "input": "Какая столица у Узбекистана?",
        "expected": "Ташкент",
        "metadata": {"lang": "ru", "topic": "geo"},
    },
    {
        "id": "qa-math-mul",
        "input": "Сколько будет 17 * 3?",
        "expected": "51",
        "metadata": {"lang": "ru", "topic": "math"},
    },
    {
        "id": "qa-math-precedence",
        "input": "What is 2 + 2 * 2?",
        "expected": "6",
        "metadata": {"lang": "en", "topic": "math", "note": "проверка приоритета операций"},
    },
    {
        "id": "qa-tolstoy",
        "input": "Кто написал «Войну и мир»?",
        "expected": "Толстой",
        "metadata": {"lang": "ru", "topic": "literature"},
    },
    {
        "id": "qa-einstein",
        "input": "Who developed the theory of relativity?",
        "expected": "Einstein",
        "metadata": {"lang": "en", "topic": "science"},
    },
    {
        "id": "qa-project-lang",
        "input": "На каком языке написан этот проект?",
        "expected": "Python",
        "metadata": {"lang": "ru", "topic": "meta"},
    },
    {
        "id": "qa-rag",
        "input": "What does RAG stand for?",
        "expected": ["Retrieval", "Generation"],
        "metadata": {"lang": "en", "topic": "ml"},
    },
    {
        "id": "qa-planets",
        "input": "Сколько планет в Солнечной системе?",
        "expected": "8",
        "metadata": {"lang": "ru", "topic": "science"},
    },
    {
        "id": "qa-boiling",
        "input": "At what temperature does water start boiling, in Celsius?",
        "expected": "100",
        "metadata": {"lang": "en", "topic": "science"},
    },
    {
        "id": "qa-mlops",
        "input": "Что такое MLOps?",
        "expected": ["эксплуатац", "машинного обучения"],
        "metadata": {"lang": "ru", "topic": "ml"},
    },
    {
        "id": "qa-nile",
        "input": "Какая самая длинная река в мире?",
        "expected": "Нил",
        "metadata": {"lang": "ru", "topic": "geo", "note": "нет в базе знаний — ожидаемый провал"},
    },
]

CONTACT_SCHEMA = {
    "type": "object",
    "required": ["name", "email", "phone", "age"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "minLength": 2},
        "email": {"type": ["string", "null"], "pattern": r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$"},
        "phone": {"type": ["string", "null"], "pattern": r"^\+?\d{9,15}$"},
        "age": {"type": ["integer", "null"], "minimum": 0, "maximum": 120},
    },
}

EXTRACTION = {
    "name": "json_extraction",
    "metadata": {"schema": CONTACT_SCHEMA},
    "cases": [
        {
            "id": "ex-ru-full",
            "input": "Здравствуйте! Меня зовут Алия Каримова, мне 29 лет. "
            "Почта aliya.karimova@example.uz, телефон +998 90 123-45-67.",
            "expected": {
                "name": "Алия Каримова",
                "email": "aliya.karimova@example.uz",
                "phone": "+998901234567",
                "age": 29,
            },
        },
        {
            "id": "ex-en-full",
            "input": "Hi, my name is John Smith, I'm 41 years old. "
            "Reach me at John.Smith@Mail.com or +1 (415) 555-0199.",
            "expected": {
                "name": "John Smith",
                "email": "john.smith@mail.com",
                "phone": "+14155550199",
                "age": 41,
            },
        },
        {
            "id": "ex-ru-no-phone",
            "input": "Пишите на hr@company.ru — меня зовут Дмитрий, 35 лет.",
            "expected": {"name": "Дмитрий", "email": "hr@company.ru", "phone": None, "age": 35},
        },
        {
            "id": "ex-en-no-age",
            "input": "Contact: Maria Garcia, maria.garcia@uni.edu, phone 8 800 555 35 35",
            "expected": {
                "name": "Maria Garcia",
                "email": "maria.garcia@uni.edu",
                "phone": "88005553535",
                "age": None,
            },
        },
        {
            "id": "ex-ru-informal",
            "input": "Я Тимур, мне 22 года, email: timur_22@mail.ru",
            "expected": {"name": "Тимур", "email": "timur_22@mail.ru", "phone": None, "age": 22},
            "metadata": {"note": "имя без маркера «меня зовут» — сложный кейс"},
        },
        {
            "id": "ex-en-obfuscated-email",
            "input": "I'm Alex Brown, 30 y.o., email alex.brown(at)gmail.com, tel +44 20 7946 0958",
            "expected": {
                "name": "Alex Brown",
                "email": "alex.brown@gmail.com",
                "phone": "+442079460958",
                "age": 30,
            },
            "metadata": {"note": "email замаскирован (at) — сложный кейс"},
        },
    ],
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    qa_path = OUT / "qa.jsonl"
    qa_path.write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in QA), encoding="utf-8"
    )
    ex_path = OUT / "json_extraction.yaml"
    ex_path.write_text(
        "# Сгенерировано scripts/make_datasets.py. Схема из metadata применяется ко всем кейсам.\n"
        + yaml.safe_dump(EXTRACTION, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )
    print(f"{qa_path}: {len(QA)} кейсов")
    print(f"{ex_path}: {len(EXTRACTION['cases'])} кейсов")


if __name__ == "__main__":
    main()
