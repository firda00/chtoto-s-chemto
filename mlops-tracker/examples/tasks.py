"""Примеры task для харнеса.

* `answer_question` — детерминированная функция: поиск по мини-базе знаний + наивный калькулятор
  (с намеренным багом: считает выражения слева направо, без приоритета операций).
* `mock_agent` — «агент» в стиле ReAct: LLM выбирает инструмент (calculator / kb) и пишет ответ.
  Без `LLM_API_KEY` политика агента — детерминированный MockLLM, так что прогон воспроизводим.
* `extract_contact` — извлечение структурированных данных (JSON) из текста регулярками.

Запуск: `uv run mlt eval run -d data/evals/qa.jsonl -t examples.tasks:answer_question -s contains`
"""

from __future__ import annotations

import ast
import json
import operator
import re
from typing import Any

from mlops_tracker.evals.harness import TaskOutput
from mlops_tracker.evals.scorers import estimate_cost
from mlops_tracker.llm import get_llm

# (ключевые слова, ответ). Слова сравниваются по префиксу, чтобы ловить падежи.
KB: list[tuple[tuple[str, ...], str]] = [
    (("capital", "france"), "The capital of France is Paris."),
    (("столица", "узбекистан"), "Столица Узбекистана — Ташкент."),
    (("войн", "мир", "написал"), "«Войну и мир» написал Лев Толстой."),
    (("theory", "relativity"), "The theory of relativity was developed by Albert Einstein."),
    (("язык", "проект"), "Этот проект написан на Python 3.12."),
    (("rag", "stand"), "RAG stands for Retrieval-Augmented Generation."),
    (("планет", "солнечн"), "В Солнечной системе 8 планет."),
    (("boiling", "water"), "Water boils at 100 °C at sea level."),
    (("mlops",), "MLOps — практики эксплуатации моделей машинного обучения в продакшене."),
]

IDK_RU = "Не знаю."
IDK_EN = "I don't know."


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _is_cyrillic(text: str) -> bool:
    return bool(re.search(r"[а-яё]", text.lower()))


def search_kb(query: str, min_hits: int = 1) -> str | None:
    """Ответ из базы знаний с максимальным числом совпавших ключевых слов (не меньше `min_hits`)."""
    words = _tokens(query)
    best, best_hits = None, 0
    for keys, answer in KB:
        hits = sum(any(w.startswith(k) for w in words) for k in keys)
        if (hits == len(keys) or hits >= 2) and hits > best_hits:
            best, best_hits = answer, hits
    return best if best_hits >= min_hits else None


_EXPR = re.compile(r"\d+(?:\.\d+)?(?:\s*[-+*/]\s*\d+(?:\.\d+)?)+")


def _naive_eval(expr: str) -> float:
    """Слева направо, без приоритета: «2 + 2 * 2» = 8. Это намеренный баг для демонстрации."""
    tokens = re.findall(r"\d+(?:\.\d+)?|[-+*/]", expr)
    acc = float(tokens[0])
    ops = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
    for op, num in zip(tokens[1::2], tokens[2::2], strict=True):
        acc = ops[op](acc, float(num))
    return acc


def _fmt_num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:.4g}"


def answer_question(question: str) -> str:
    """Детерминированный бейзлайн: калькулятор → база знаний → «не знаю»."""
    m = _EXPR.search(question)
    if m:
        return f"Ответ: {_fmt_num(_naive_eval(m.group(0)))}"
    found = search_kb(question)
    if found:
        return found
    return IDK_RU if _is_cyrillic(question) else IDK_EN


# ------------------------------------------------------------------ agent
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def calculator(expr: str) -> str:
    """Безопасный калькулятор на AST (корректный приоритет операций)."""

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError(f"недопустимое выражение: {expr}")

    return _fmt_num(ev(ast.parse(expr, mode="eval").body))


TOOLS = {
    "calculator": calculator,
    # У агента более строгий ретривер: нужно ≥ 2 совпавших ключевых слова.
    "kb": lambda q: search_kb(q, min_hits=2) or "NOT_FOUND",
}

ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["calculator", "kb", "final"]},
        "input": {"type": "string"},
    },
    "required": ["action", "input"],
}

AGENT_SYSTEM = (
    "Ты — агент. Доступные инструменты: calculator(expr), kb(query). "
    'На каждом шаге верни JSON {"action": "calculator|kb|final", "input": "..."}.'
)


def mock_policy(prompt: str, system: str | None, json_schema: dict[str, Any] | None) -> dict:
    """Детерминированная «модель» для MockLLM: читает вопрос и наблюдения из промпта."""
    question = re.search(r"Вопрос: (.*)", prompt).group(1)  # type: ignore[union-attr]
    observations = re.findall(r"Наблюдение: (.*)", prompt)
    ru = _is_cyrillic(question)
    if observations:
        obs = observations[-1]
        if obs == "NOT_FOUND":
            return {"action": "final", "input": IDK_RU if ru else IDK_EN}
        if re.fullmatch(r"-?[\d.]+", obs):
            return {"action": "final", "input": f"Ответ: {obs}" if ru else f"Answer: {obs}"}
        return {"action": "final", "input": obs}
    m = _EXPR.search(question)
    if m:
        return {"action": "calculator", "input": m.group(0)}
    return {"action": "kb", "input": question}


def mock_agent(question: str, max_steps: int = 4) -> TaskOutput:
    """ReAct-агент: шаги «решение → инструмент → наблюдение», с учётом токенов и стоимости."""
    llm = get_llm(mock_handler=mock_policy)
    scratchpad: list[str] = [f"Вопрос: {question}"]
    trace: list[dict[str, Any]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    answer = IDK_RU if _is_cyrillic(question) else IDK_EN
    for _ in range(max_steps):
        decision = llm.complete(
            "\n".join(scratchpad), system=AGENT_SYSTEM, json_schema=ACTION_SCHEMA
        )
        if isinstance(decision, str):
            decision = json.loads(decision)
        for k in usage:
            usage[k] += llm.last_usage.get(k, 0)
        action, arg = decision.get("action"), str(decision.get("input", ""))
        trace.append({"action": action, "input": arg})
        if action == "final":
            answer = arg
            break
        tool = TOOLS.get(action or "")
        try:
            observation = tool(arg) if tool else f"неизвестный инструмент {action}"
        except Exception as e:  # noqa: BLE001 — ошибка инструмента = наблюдение для агента
            observation = f"ошибка: {e}"
        trace[-1]["observation"] = observation
        scratchpad.append(f"Действие: {action}({arg})")
        scratchpad.append(f"Наблюдение: {observation}")
    return TaskOutput(
        output=answer,
        usage=usage,
        cost_usd=estimate_cost(usage),
        metadata={"steps": len(trace), "trace": trace, "model": llm.model},
    )


# ------------------------------------------------------------- extraction
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"\+?\d[\d\s()-]{8,}\d")
_AGE = re.compile(r"(\d{1,3})\s*(?:лет|года|год|years?\s*old|y\.?o\.?)", re.I)
_NAME = re.compile(
    r"\b(?i:меня зовут|my name is|i am|i'm|contact:?)\s+"
    r"([A-ZА-ЯЁ][a-zа-яё]+(?:\s+[A-ZА-ЯЁ][a-zа-яё]+)?)"
)


def extract_contact(text: str) -> str:
    """Извлекает {name, email, phone, age} и возвращает JSON-строку (как ответила бы LLM)."""
    name = _NAME.search(text)
    email = _EMAIL.search(text)
    phone = _PHONE.search(text)
    age = _AGE.search(text)
    data = {
        "name": name.group(1) if name else None,
        "email": email.group(0).lower() if email else None,
        "phone": re.sub(r"[^\d+]", "", phone.group(0)) if phone else None,
        "age": int(age.group(1)) if age else None,
    }
    return json.dumps(data, ensure_ascii=False)
