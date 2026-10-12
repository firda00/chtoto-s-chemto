from datetime import datetime

import pytest

from rag_system.llm import AssistantMessage, MockLLM, ToolCall
from rag_system.tools import default_registry, run_tool_loop, safe_eval


@pytest.mark.parametrize(
    ("expr", "value"),
    [
        ("2 + 2 * 3", 8),
        ("(1 + 2) ** 3", 27),
        ("120 000 * 3", 360000),
        ("7 / 2", 3.5),
        ("-5 % 3", 1),
        ("1 400 000 × 0.9", 1260000),
    ],
)
def test_calculator(expr: str, value: float) -> None:
    assert safe_eval(expr) == value


@pytest.mark.parametrize("expr", ["__import__('os')", "a + 1", "2 ** 1000", "[1, 2]"])
def test_calculator_rejects_unsafe(expr: str) -> None:
    with pytest.raises((ValueError, SyntaxError)):
        safe_eval(expr)


def _clock() -> datetime:
    return datetime(2026, 10, 12, 9, 30)


def test_tool_loop_with_scripted_calls() -> None:
    llm = MockLLM(
        [
            {"tool_calls": [{"name": "calculator", "arguments": {"expression": "150000 * 3"}}]},
            AssistantMessage(
                tool_calls=[ToolCall(id="d1", name="get_date", arguments={})],
            ),
            "Три часа в «Луне» стоят 450000 сум; сегодня 2026-10-12.",
        ]
    )
    registry = default_registry(now=_clock)
    messages = [{"role": "user", "content": "Сколько стоят 3 часа Луны и какое сегодня число?"}]
    result = run_tool_loop(llm, messages, registry, max_steps=4)

    assert result.content.startswith("Три часа")
    assert [t.name for t in result.trace] == ["calculator", "get_date"]
    assert result.trace[0].result == "450000"
    assert '"date": "2026-10-12"' in result.trace[1].result
    assert result.steps == 3
    # история сообщений в формате OpenAI: assistant(tool_calls) → tool
    roles = [m["role"] for m in result.messages]
    assert roles == ["user", "assistant", "tool", "assistant", "tool"]
    assert result.messages[4]["tool_call_id"] == "d1"
    # инструменты передавались в LLM в формате function calling
    tools = llm.calls[0]["tools"]
    assert {t["function"]["name"] for t in tools} == {"calculator", "get_date"}
    assert messages == [messages[0]]  # исходный список не мутируется


def test_tool_loop_handles_errors_and_step_limit() -> None:
    always_call = {"tool_calls": [{"name": "nope", "arguments": {}}]}
    llm = MockLLM([always_call, always_call, "final"])
    result = run_tool_loop(llm, [{"role": "user", "content": "x"}], default_registry(), 2)
    assert result.content == "final"
    assert result.trace[0].result.startswith("error: unknown tool")
    assert llm.calls[-1]["tools"] is None  # финальный вызов без инструментов


def test_tool_error_is_returned_to_model() -> None:
    reg = default_registry()
    assert reg.call("calculator", {"expression": "1/0"}).startswith("error: ZeroDivisionError")
    assert reg.call("calculator", {}).startswith("error: TypeError")


def test_mock_default_chat_calls_calculator() -> None:
    llm = MockLLM()
    messages = [{"role": "user", "content": "<question>Сколько будет 12 * 4?</question>"}]
    result = run_tool_loop(llm, messages, default_registry(include={"calculator"}))
    assert result.trace[0].name == "calculator"
    assert "48" in result.content
