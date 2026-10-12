import json

import httpx
import pytest
import respx
from pydantic import BaseModel

from local_llm import llm
from local_llm.client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    LLMClient,
    LLMSettings,
    MockLLM,
    get_llm,
    parse_json_answer,
)


class Verdict(BaseModel):
    label: str
    score: float
    tags: list[str]


def test_settings_from_env(monkeypatch):
    for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    s = LLMSettings.from_env()
    assert (s.base_url, s.model, s.api_key) == (DEFAULT_BASE_URL, DEFAULT_MODEL, "")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen3:8b")
    monkeypatch.setenv("LLM_API_KEY", "k")
    s = LLMSettings.from_env(model="override")
    assert s.base_url.endswith(":11434/v1") and s.model == "override" and s.api_key == "k"
    assert "k" not in repr(s)


def test_mock_llm_is_deterministic():
    m = MockLLM(model="m")
    assert m.complete("привет") == m.complete("привет")
    assert m.complete("привет") != m.complete("пока")
    assert MockLLM(responses={"q": "a"}).complete("q") == "a"
    assert len(m.calls) == 4


def test_mock_llm_json_schema():
    out = MockLLM().complete("x", json_schema=Verdict)
    assert Verdict.model_validate(out) == Verdict(label="mock", score=0.0, tags=[])
    schema = {
        "type": "object",
        "properties": {"kind": {"enum": ["a", "b"]}, "ok": {"type": "boolean"}},
    }
    assert MockLLM().complete("x", json_schema=schema) == {"kind": "a", "ok": False}


def test_get_llm_respects_env(monkeypatch):
    monkeypatch.setenv("LLM_MOCK", "1")
    assert isinstance(get_llm(), MockLLM)
    assert isinstance(llm.get_llm(mock=False), LLMClient)
    monkeypatch.setenv("LLM_MOCK", "0")
    assert isinstance(get_llm(), LLMClient)


def test_module_level_complete_uses_mock(monkeypatch):
    monkeypatch.setenv("LLM_MOCK", "true")
    assert llm.complete("hi").startswith("[")


def test_parse_json_answer_handles_fences():
    assert parse_json_answer('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_answer('Вот ответ: {"a": 2} — готово') == {"a": 2}
    with pytest.raises(json.JSONDecodeError):
        parse_json_answer("нет json")


def _completion(content: str) -> dict:
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


@respx.mock
def test_llm_client_complete_text_and_json():
    route = respx.post("http://local.test/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(200, json=_completion("ответ")),
            httpx.Response(
                200, json=_completion('```json\n{"label":"ok","score":1,"tags":[]}\n```')
            ),
        ]
    )
    client = LLMClient(LLMSettings(base_url="http://local.test/v1", model="qwen3:8b"))
    assert client.complete("q", system="sys") == "ответ"
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "qwen3:8b"
    assert body["messages"][0] == {"role": "system", "content": "sys"}

    out = client.complete("q", json_schema=Verdict)
    assert out == {"label": "ok", "score": 1, "tags": []}
    body = json.loads(route.calls[1].request.content)
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"]["title"] == "Verdict"


@respx.mock
def test_llm_client_stream():
    from local_llm.mock_server import FakeClock, make_handler

    respx.post("http://local.test/v1/chat/completions").mock(
        side_effect=make_handler(clock=FakeClock())
    )
    client = LLMClient(LLMSettings(base_url="http://local.test/v1", model="m"))
    text = "".join(client.stream("hello"))
    assert text == MockLLM(model="m").answer_text("hello", n_words=63)
