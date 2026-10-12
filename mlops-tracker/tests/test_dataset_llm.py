from __future__ import annotations

from pathlib import Path

import pytest

from mlops_tracker.evals.dataset import Dataset, load_dataset
from mlops_tracker.llm import MockLLM, OpenAICompatibleLLM, extract_json, get_llm

DATA = Path(__file__).resolve().parents[1] / "data" / "evals"


def test_load_demo_datasets() -> None:
    qa = load_dataset(DATA / "qa.jsonl")
    assert len(qa) == 12
    assert {c.metadata["lang"] for c in qa} == {"ru", "en"}
    ex = load_dataset(DATA / "json_extraction.yaml")
    assert ex.name == "json_extraction" and len(ex) == 6
    assert all(c.metadata["schema"]["type"] == "object" for c in ex)  # схема из верхнего уровня
    assert "note" in ex.cases[-1].metadata
    assert len(qa.hash) == 12 and qa.hash != ex.hash


def test_jsonl_comments_auto_ids_and_errors(tmp_path: Path) -> None:
    p = tmp_path / "d.jsonl"
    p.write_text('# комментарий\n{"input": "a"}\n\n{"input": "b", "expected": 1}\n', "utf-8")
    ds = load_dataset(p)
    assert [c.id for c in ds] == ["case-1", "case-2"]
    p.write_text('{"input": 1}\n{oops\n', "utf-8")
    with pytest.raises(ValueError, match=":2"):
        load_dataset(p)
    with pytest.raises(ValueError, match="Повторяющийся"):
        Dataset.from_records([{"id": 1, "input": "a"}, {"id": "1", "input": "b"}])
    with pytest.raises(ValueError, match="input"):
        Dataset.from_records([{"expected": "x"}])
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.jsonl")
    (tmp_path / "x.csv").write_text("a,b", "utf-8")
    with pytest.raises(ValueError, match="формат"):
        load_dataset(tmp_path / "x.csv")


def test_hash_changes_with_content() -> None:
    a = Dataset.from_records([{"id": "1", "input": "x", "expected": "y"}])
    b = Dataset.from_records([{"id": "1", "input": "x", "expected": "z"}])
    assert a.hash != b.hash


def test_extract_json_variants() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Вот ответ: {"a": [1, 2]} — готово') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        extract_json("нет json")


def test_mock_llm_modes_and_usage() -> None:
    llm = MockLLM(responses=["первый"])
    assert llm.complete("привет") == "первый"
    assert llm.complete("привет мир").startswith("mock:")
    assert llm.last_usage["prompt_tokens"] == 2
    schema = {"type": "object", "properties": {"n": {"type": "integer"}, "s": {"type": "string"}}}
    assert llm.complete("x", json_schema=schema) == {"n": 0, "s": ""}
    handled = MockLLM(handler=lambda p, s, j: '{"ok": true}')
    assert handled.complete("x", json_schema={"type": "object"}) == {"ok": True}
    assert len(llm.calls) == 3


def test_get_llm_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(get_llm(), MockLLM)
    monkeypatch.setenv("LLM_PROVIDER", "auto")
    assert isinstance(get_llm(), MockLLM)  # ключа нет → заглушка
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    assert isinstance(get_llm(), OpenAICompatibleLLM)  # клиент создаётся без сетевых вызовов
    assert isinstance(get_llm("mock"), MockLLM)
    with pytest.raises(ValueError):
        get_llm("nonsense")
