from __future__ import annotations

import pytest

from mlops_tracker.evals.dataset import Case
from mlops_tracker.evals.scorers import (
    SCORERS,
    Contains,
    Cost,
    ExactMatch,
    JsonMatch,
    JsonSchema,
    Latency,
    LLMJudge,
    NumericTolerance,
    Regex,
    Score,
    ScoreContext,
    apply_scorer,
    build_scorer,
    build_scorers,
    validate_json_schema,
)
from mlops_tracker.llm import MockLLM

CTX = ScoreContext()


def case(expected=None, **metadata) -> Case:
    return Case(id="t", input="вопрос", expected=expected, metadata=metadata)


def test_exact_match_normalizes_whitespace_and_case() -> None:
    s = ExactMatch()
    assert s(case("Paris"), "  paris ", CTX).passed
    assert s(case(["Париж", "Paris"]), "ПАРИЖ", CTX).passed
    assert not s(case("Paris"), "Paris, France", CTX).passed
    assert not ExactMatch(case_sensitive=True)(case("Paris"), "paris", CTX).passed


def test_contains_all_and_any() -> None:
    s = Contains()
    res = s(case(["Retrieval", "Generation"]), "RAG = retrieval-augmented generation", CTX)
    assert res.passed and res.value == 1.0
    partial = s(case(["Retrieval", "Rocket"]), "retrieval", CTX)
    assert not partial.passed and partial.value == 0.5
    assert partial.details["missing"] == ["Rocket"]
    assert s(case(["нет", "Ташкент"], contains_mode="any"), "Ташкент", CTX).passed


def test_regex_from_metadata_or_expected() -> None:
    assert Regex()(case(regex=r"\b\d{4}\b"), "в 2008 году", CTX).passed
    assert Regex()(case(r"^ok$"), "OK", CTX).passed  # IGNORECASE по умолчанию
    assert not Regex()(case(r"^\d+$"), "abc", CTX).passed


SCHEMA = {
    "type": "object",
    "required": ["name", "age"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "minLength": 2},
        "age": {"type": ["integer", "null"], "minimum": 0, "maximum": 120},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
    },
}


def test_json_schema_scorer_valid_and_invalid() -> None:
    s = JsonSchema()
    assert s(case(schema=SCHEMA), '```json\n{"name": "Ann", "age": 30}\n```', CTX).passed
    assert s(case(schema=SCHEMA), {"name": "Ann", "age": None}, CTX).passed
    bad = s(case(schema=SCHEMA), '{"name": "A", "age": 200, "x": 1}', CTX)
    assert not bad.passed
    errors = " ".join(bad.details["errors"])
    assert "короче" in errors and "максимума" in errors and "лишнее поле" in errors
    assert not s(case(schema=SCHEMA), "не JSON вообще", CTX).passed
    assert not s(case(), "{}", CTX).passed  # нет схемы


def test_validate_json_schema_types_and_items() -> None:
    assert validate_json_schema({"name": "Ann", "age": 1, "tags": ["a"]}, SCHEMA) == []
    errs = validate_json_schema({"name": "Ann", "age": True, "tags": ["a", 1, "c"]}, SCHEMA)
    assert any("$.age" in e for e in errs)
    assert any("$.tags[1]" in e for e in errs)
    assert any("элементов больше" in e for e in errs)
    assert validate_json_schema("x", {"enum": ["a", "b"]})


def test_json_match_partial() -> None:
    res = JsonMatch()(case({"name": "Ann", "age": 30}), '{"name": "Ann", "age": 31}', CTX)
    assert res.value == 0.5 and not res.passed
    assert res.details["mismatched"]["age"] == {"expected": 30, "got": 31}


def test_numeric_tolerance() -> None:
    assert NumericTolerance()(case("51"), "Ответ: 51", CTX).passed
    assert NumericTolerance(tol=0.01)(case(3.14159), "≈ 3,14", CTX).passed
    assert not NumericTolerance(tol=0.001)(case(3.14159), "3.14", CTX).passed
    assert NumericTolerance(tol=0.1, relative=True)(case(100), "105", CTX).passed
    assert not NumericTolerance()(case("6"), "нет числа", CTX).passed


def test_llm_judge_with_mock_responses_and_prompt() -> None:
    llm = MockLLM(responses=[{"score": 5, "reason": "идеально"}, '{"score": 2, "reason": "мимо"}'])
    judge = LLMJudge(llm=llm, threshold=4)
    good = judge(case("Paris"), "Paris", CTX)
    assert good.passed and good.value == 1.0 and good.details["judge_score"] == 5
    bad = judge(case("Paris"), "London", CTX)
    assert not bad.passed and bad.value == 0.25
    prompt = llm.calls[0]["prompt"]
    assert "<expected>Paris</expected>" in prompt and "<output>Paris</output>" in prompt
    assert llm.calls[0]["json_schema"]["required"] == ["score", "reason"]


def test_llm_judge_default_mock_heuristic_and_bad_output() -> None:
    judge = LLMJudge(llm=MockLLM())
    assert judge(case("Ташкент"), "Столица — Ташкент.", CTX).details["judge_score"] == 5
    assert judge(case("Нил"), "Не знаю.", CTX).details["judge_score"] == 1
    custom = LLMJudge(llm=MockLLM(handler=lambda p, s, j: "оценка: 4 из 5"))
    assert custom(case("x"), "x", CTX).details["judge_score"] == 4
    garbage = LLMJudge(llm=MockLLM(handler=lambda p, s, j: "без оценки"))
    assert not garbage(case("x"), "x", CTX).passed


def test_latency_and_cost() -> None:
    assert Latency(max_ms=100)(case(), "x", ScoreContext(latency_ms=50)).passed
    slow = Latency(max_ms=100)(case(), "x", ScoreContext(latency_ms=400))
    assert not slow.passed and slow.value == 0.25
    usage = {"prompt_tokens": 1000, "completion_tokens": 1000}
    cheap = Cost(max_usd=0.01, price_in_per_1k=0.001, price_out_per_1k=0.002)
    res = cheap(case(), "x", ScoreContext(usage=usage))
    assert res.passed and res.details["cost_usd"] == pytest.approx(0.003)
    assert not Cost(max_usd=0.001)(case(), "x", ScoreContext(cost_usd=0.01)).passed


def test_registry_and_specs() -> None:
    assert set(SCORERS) >= {
        "exact_match",
        "contains",
        "regex",
        "json_schema",
        "numeric_tolerance",
        "llm_judge",
        "latency",
        "cost",
    }
    assert build_scorer("latency:2000").max_ms == 2000
    assert build_scorer("numeric_tolerance:0.5").tol == 0.5
    llm = MockLLM()
    judge = build_scorer("llm_judge:3", llm=llm)
    assert judge.threshold == 3 and judge.llm is llm
    assert [s.name for s in build_scorers("contains, exact_match")] == ["contains", "exact_match"]
    with pytest.raises(ValueError):
        build_scorer("unknown")
    with pytest.raises(ValueError):
        build_scorer("contains:1")


def test_apply_scorer_accepts_custom_functions() -> None:
    def short(case, output):
        return len(output) < 10

    def ratio(case, output, ctx):
        return 0.7

    def broken(case, output):
        raise RuntimeError("bug")

    assert apply_scorer(short, case(), "ok", CTX) == Score(name="short", value=1.0, passed=True)
    r = apply_scorer(ratio, case(), "ok", CTX)
    assert r.name == "ratio" and r.passed and r.value == 0.7
    err = apply_scorer(broken, case(), "ok", CTX)
    assert not err.passed and "bug" in err.details["error"]
