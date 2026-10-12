"""Генерирует data/prompts.txt — набор промптов для бенчмарка (RU/EN, разной длины ответа).

Запуск: uv run python scripts/make_prompts.py
"""

from pathlib import Path

PROMPTS = [
    # короткие ответы — TTFT доминирует
    "Переведи на английский: «Квантование уменьшает потребление памяти модели».",
    "What is the capital of Uzbekistan? Answer in one word.",
    "Классифицируй тональность отзыва (позитив/негатив/нейтрально): «Доставка задержалась, но товар отличный».",
    # средние — баланс TTFT и декодинга
    "Объясни простыми словами, что такое KV-кэш в трансформерах и почему он растёт с длиной контекста.",
    "Explain the difference between AWQ and GPTQ quantization in 5 bullet points.",
    "Напиши функцию на Python, которая возвращает n-е число Фибоначчи итеративно, с docstring.",
    "Summarize the trade-offs between latency and throughput when serving LLMs with batching.",
    # длинные — декодинг доминирует
    "Составь подробный план миграции сервиса с облачного LLM API на локальный vLLM: этапы, риски, метрики.",
    "Write a short technical blog post (about 200 words) on why tokens per second matters for chat UX.",
    "Сравни Ollama и vLLM для продакшена: производительность, удобство, квантование, мульти-GPU. Ответ таблицей.",
]


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data" / "prompts.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = "# Промпты для `uv run llm bench`: один на строку, строки с # игнорируются\n"
    out.write_text(header + "\n".join(PROMPTS) + "\n", encoding="utf-8")
    print(f"data/prompts.txt: {len(PROMPTS)} prompts")


if __name__ == "__main__":
    main()
