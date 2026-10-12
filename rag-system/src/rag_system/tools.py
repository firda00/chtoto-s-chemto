"""Инструменты для OpenAI-style function calling и цикл вызова инструментов.

Цикл: LLM.chat(messages, tools) → если есть tool_calls, выполняем каждый, добавляем
сообщения role=tool и повторяем; если tool_calls нет — это финальный ответ.
После max_steps делаем финальный вызов без инструментов, чтобы гарантированно получить текст.
"""

from __future__ import annotations

import ast
import json
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rag_system.llm import LLM
from rag_system.schemas import ToolCallRecord


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> ToolRegistry:
        self.tools[tool.name] = tool
        return self

    def __contains__(self, name: str) -> bool:
        return name in self.tools

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self.tools.values()]

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"error: unknown tool {name!r}"
        try:
            result = tool.func(**arguments)
        except Exception as exc:  # ошибка инструмента возвращается модели, а не роняет пайплайн
            return f"error: {type(exc).__name__}: {exc}"
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


# --------------------------------------------------------------------------------------------
# calculator: безопасный eval через AST (никакого eval/exec)
# --------------------------------------------------------------------------------------------
_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval(expression: str) -> float | int:
    expr = expression.replace("×", "*").replace("^", "**").replace(" ", " ")
    # «1 400 000» → 1400000; запятая как десятичный разделитель
    expr = re.sub(r"(?<=\d) (?=\d{3}\b)", "", expr).replace(",", ".")
    if len(expr) > 200:
        raise ValueError("expression too long")
    tree = ast.parse(expr, mode="eval")

    def ev(node: ast.AST) -> float | int:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression element: {type(node).__name__}")

    value = ev(tree)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return round(value, 10) if isinstance(value, float) else value


def calculator_tool() -> Tool:
    return Tool(
        name="calculator",
        description="Evaluate an arithmetic expression (+ - * / // % ** and parentheses).",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "e.g. 120000 * 3"}},
            "required": ["expression"],
        },
        func=lambda expression: str(safe_eval(expression)),
    )


def date_tool(now: Callable[[], datetime] | None = None) -> Tool:
    clock = now or datetime.now

    def get_date() -> dict[str, str]:
        d = clock()
        return {
            "date": d.date().isoformat(),
            "weekday": d.strftime("%A"),
            "time": d.strftime("%H:%M"),
        }

    return Tool(
        name="get_date",
        description="Get the current local date, weekday and time.",
        parameters={"type": "object", "properties": {}},
        func=get_date,
    )


def search_docs_tool(search: Callable[[str, str | None], str]) -> Tool:
    return Tool(
        name="search_docs",
        description=(
            "Search the company knowledge base (hybrid BM25 + vector). Returns numbered "
            "fragments that can be cited as [n]."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search query"},
                "collection": {
                    "type": "string",
                    "description": "optional collection name to restrict the search",
                },
            },
            "required": ["query"],
        },
        func=lambda query, collection=None: search(query, collection),
    )


def default_registry(
    search: Callable[[str, str | None], str] | None = None,
    now: Callable[[], datetime] | None = None,
    include: set[str] | None = None,
) -> ToolRegistry:
    reg = ToolRegistry()
    if search is not None:
        reg.register(search_docs_tool(search))
    if include is None or "calculator" in include:
        reg.register(calculator_tool())
    if include is None or "get_date" in include:
        reg.register(date_tool(now))
    return reg


# --------------------------------------------------------------------------------------------
# tool loop
# --------------------------------------------------------------------------------------------
@dataclass
class ToolLoopResult:
    content: str
    trace: list[ToolCallRecord]
    messages: list[dict[str, Any]]
    steps: int


def run_tool_loop(
    llm: LLM, messages: list[dict[str, Any]], registry: ToolRegistry, max_steps: int = 4
) -> ToolLoopResult:
    messages = list(messages)
    trace: list[ToolCallRecord] = []
    specs = registry.specs() or None
    for step in range(1, max_steps + 1):
        reply = llm.chat(messages, tools=specs)
        if not reply.tool_calls:
            return ToolLoopResult(reply.content or "", trace, messages, step)
        messages.append(
            {
                "role": "assistant",
                "content": reply.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in reply.tool_calls
                ],
            }
        )
        for tc in reply.tool_calls:
            result = registry.call(tc.name, tc.arguments)
            trace.append(ToolCallRecord(name=tc.name, arguments=tc.arguments, result=result))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    # лимит шагов исчерпан → просим финальный ответ без инструментов
    final = llm.chat(messages, tools=None)
    return ToolLoopResult(final.content or "", trace, messages, max_steps + 1)
