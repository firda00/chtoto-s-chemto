"""Скореры: функции `(case, output, ctx) -> Score`.

Встроенные: exact_match, contains, regex, json_schema, json_match, numeric_tolerance,
llm_judge, latency, cost. Своим скорером может быть любая функция с той же сигнатурой,
возвращающая `Score`, `bool` или число 0..1.
"""

from __future__ import annotations

import inspect
import json
import math
import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from mlops_tracker.evals.dataset import Case
from mlops_tracker.llm import LLM, extract_json, get_llm


class Score(BaseModel):
    name: str
    value: float = Field(description="Нормированная оценка 0..1")
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


class ScoreContext(BaseModel):
    """Всё, что харнес знает о выполнении кейса, кроме самого ответа."""

    latency_ms: float = 0.0
    usage: dict[str, int] = Field(default_factory=dict)
    cost_usd: float | None = None
    attempts: int = 1
    error: str | None = None


class Scorer:
    name: str = "scorer"

    def __call__(self, case: Case, output: Any, ctx: ScoreContext) -> Score:
        return self.score(case, output, ctx)

    def score(self, case: Case, output: Any, ctx: ScoreContext) -> Score:
        raise NotImplementedError

    def _s(self, value: float, passed: bool | None = None, **details: Any) -> Score:
        value = max(0.0, min(1.0, float(value)))
        return Score(
            name=self.name,
            value=value,
            passed=value >= 1.0 if passed is None else passed,
            details=details,
        )


def as_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False)


def _norm(text: str, case_sensitive: bool) -> str:
    text = " ".join(text.strip().split())
    return text if case_sensitive else text.casefold()


def _as_list(x: Any) -> list[Any]:
    if x is None:
        return []
    return list(x) if isinstance(x, list | tuple) else [x]


# --------------------------------------------------------------------- text
class ExactMatch(Scorer):
    """Совпадение после нормализации пробелов (и регистра). `expected` может быть списком
    допустимых вариантов."""

    name = "exact_match"

    def __init__(self, case_sensitive: bool = False) -> None:
        self.case_sensitive = case_sensitive

    def score(self, case: Case, output: Any, ctx: ScoreContext) -> Score:
        out = _norm(as_text(output), self.case_sensitive)
        options = [_norm(as_text(e), self.case_sensitive) for e in _as_list(case.expected)]
        return self._s(1.0 if out in options else 0.0)


class Contains(Scorer):
    """Все подстроки из `expected` (строка или список) входят в ответ. value = доля найденных.
    Режим «любая из» — `metadata.contains_mode: any`."""

    name = "contains"

    def __init__(self, case_sensitive: bool = False) -> None:
        self.case_sensitive = case_sensitive

    def score(self, case: Case, output: Any, ctx: ScoreContext) -> Score:
        out = _norm(as_text(output), self.case_sensitive)
        needles = _as_list(case.metadata.get("contains", case.expected))
        if not needles:
            return self._s(0.0, reason="нет ожидаемых подстрок")
        found = [n for n in needles if _norm(as_text(n), self.case_sensitive) in out]
        missing = [n for n in needles if n not in found]
        if case.metadata.get("contains_mode") == "any":
            return self._s(1.0 if found else 0.0, found=found, missing=missing)
        return self._s(len(found) / len(needles), found=found, missing=missing)


class Regex(Scorer):
    """`re.search` по шаблону из `metadata.regex` (или из `expected`)."""

    name = "regex"

    def __init__(self, pattern: str | None = None, flags: int = re.IGNORECASE) -> None:
        self.pattern = pattern
        self.flags = flags

    def score(self, case: Case, output: Any, ctx: ScoreContext) -> Score:
        pattern = self.pattern or case.metadata.get("regex") or as_text(case.expected)
        m = re.search(pattern, as_text(output), self.flags)
        return self._s(1.0 if m else 0.0, pattern=pattern, match=m.group(0) if m else None)


# --------------------------------------------------------------------- JSON
_TYPES: dict[str, Callable[[Any], bool]] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def validate_json_schema(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Мини-валидатор подмножества JSON Schema (без внешних зависимостей):
    type, enum, const, required, properties, additionalProperties, items, minItems, maxItems,
    minLength, maxLength, pattern, minimum, maximum. Возвращает список ошибок."""
    errors: list[str] = []
    types = _as_list(schema.get("type"))
    if types and not any(_TYPES.get(t, lambda _: False)(value) for t in types):
        return [f"{path}: ожидался тип {'|'.join(types)}, получено {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} не входит в {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: ожидалось {schema['const']!r}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: нет обязательного поля '{key}'")
        for key, sub in value.items():
            if key in props:
                errors.extend(validate_json_schema(sub, props[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: лишнее поле '{key}'")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: элементов меньше {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: элементов больше {schema['maxItems']}")
        if isinstance(schema.get("items"), dict):
            for i, item in enumerate(value):
                errors.extend(validate_json_schema(item, schema["items"], f"{path}[{i}]"))
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: строка короче {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: строка длиннее {schema['maxLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: не соответствует шаблону {schema['pattern']}")
    if _TYPES["number"](value):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: меньше минимума {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: больше максимума {schema['maximum']}")
    return errors


def parse_json_output(output: Any) -> Any:
    if isinstance(output, dict | list):
        return output
    return extract_json(as_text(output))


class JsonSchema(Scorer):
    """Ответ — валидный JSON, соответствующий схеме из `metadata.schema` (или переданной явно)."""

    name = "json_schema"

    def __init__(self, schema: dict[str, Any] | None = None) -> None:
        self.schema = schema

    def score(self, case: Case, output: Any, ctx: ScoreContext) -> Score:
        schema = self.schema or case.metadata.get("schema")
        if not schema:
            return self._s(0.0, errors=["у кейса нет metadata.schema"])
        try:
            data = parse_json_output(output)
        except ValueError as e:
            return self._s(0.0, errors=[str(e)])
        errors = validate_json_schema(data, schema)
        return self._s(0.0 if errors else 1.0, errors=errors)


class JsonMatch(Scorer):
    """Поля из `expected` (dict) совпадают с полями ответа. value = доля совпавших полей."""

    name = "json_match"

    def score(self, case: Case, output: Any, ctx: ScoreContext) -> Score:
        if not isinstance(case.expected, dict) or not case.expected:
            return self._s(0.0, reason="expected должен быть непустым объектом")
        try:
            data = parse_json_output(output)
        except ValueError as e:
            return self._s(0.0, reason=str(e))
        if not isinstance(data, dict):
            return self._s(0.0, reason="ответ — не объект")
        mismatched = {
            k: {"expected": v, "got": data.get(k)}
            for k, v in case.expected.items()
            if data.get(k) != v
        }
        ok = len(case.expected) - len(mismatched)
        return self._s(ok / len(case.expected), mismatched=mismatched)


# ------------------------------------------------------------------ numeric
_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


class NumericTolerance(Scorer):
    """Первое число в ответе отличается от `expected` не больше чем на `tol`
    (абсолютно или относительно при `relative=True`)."""

    name = "numeric_tolerance"

    def __init__(self, tol: float = 1e-6, relative: bool = False) -> None:
        self.tol = float(tol)
        self.relative = relative

    def score(self, case: Case, output: Any, ctx: ScoreContext) -> Score:
        try:
            expected = float(str(case.expected).replace(",", "."))
        except (TypeError, ValueError):
            return self._s(0.0, reason=f"expected не число: {case.expected!r}")
        if isinstance(output, int | float) and not isinstance(output, bool):
            got = float(output)
        else:
            m = _NUM.search(as_text(output))
            if not m:
                return self._s(0.0, reason="в ответе нет числа")
            got = float(m.group(0).replace(",", "."))
        diff = abs(got - expected)
        limit = self.tol * abs(expected) if self.relative else self.tol
        ok = diff <= limit or math.isclose(got, expected)
        return self._s(1.0 if ok else 0.0, got=got, expected=expected, diff=diff)


# ---------------------------------------------------------------- LLM judge
DEFAULT_RUBRIC = """Оцени ответ ассистента по шкале 1–5:
5 — полностью корректен и совпадает по смыслу с эталоном;
4 — корректен, есть несущественные неточности или лишние детали;
3 — частично корректен, упущена важная часть;
2 — в основном неверен, но есть релевантные элементы;
1 — неверен, не по теме или пуст."""

JUDGE_SYSTEM = "Ты — строгий и беспристрастный эксперт по оценке ответов ИИ-ассистентов."

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "reason": {"type": "string"},
    },
    "required": ["score", "reason"],
}


def build_judge_prompt(case: Case, output: Any, rubric: str) -> str:
    return (
        f"{rubric}\n\n"
        f"<input>{as_text(case.input)}</input>\n"
        f"<expected>{as_text(case.expected)}</expected>\n"
        f"<output>{as_text(output)}</output>\n\n"
        'Верни JSON: {"score": <1-5>, "reason": "<коротко почему>"}'
    )


class LLMJudge(Scorer):
    """LLM-as-a-judge по рубрике: оценка 1–5 → value = (score-1)/4, passed = score >= threshold.
    Рубрику можно переопределить на уровне кейса через `metadata.rubric`."""

    name = "llm_judge"

    def __init__(
        self, llm: LLM | None = None, rubric: str = DEFAULT_RUBRIC, threshold: int = 4
    ) -> None:
        self.llm = llm
        self.rubric = rubric
        self.threshold = int(threshold)

    def score(self, case: Case, output: Any, ctx: ScoreContext) -> Score:
        if self.llm is None:
            self.llm = get_llm()
        prompt = build_judge_prompt(case, output, case.metadata.get("rubric", self.rubric))
        raw = self.llm.complete(prompt, system=JUDGE_SYSTEM, json_schema=JUDGE_SCHEMA)
        try:
            data = raw if isinstance(raw, dict) else extract_json(raw)
            score = int(data["score"])
            reason = str(data.get("reason", ""))
        except (ValueError, KeyError, TypeError):
            m = re.search(r"[1-5]", as_text(raw))
            if not m:
                return self._s(0.0, False, error=f"не удалось разобрать ответ судьи: {raw!r}")
            score, reason = int(m.group(0)), as_text(raw)
        score = max(1, min(5, score))
        return self._s(
            (score - 1) / 4,
            score >= self.threshold,
            judge_score=score,
            reason=reason,
            judge_model=getattr(self.llm, "model", "?"),
        )


# ------------------------------------------------------ operational metrics
class Latency(Scorer):
    """Время кейса не больше `max_ms`. value = 1, если уложились, иначе max_ms / latency."""

    name = "latency"

    def __init__(self, max_ms: float = 5000.0) -> None:
        self.max_ms = float(max_ms)

    def score(self, case: Case, output: Any, ctx: ScoreContext) -> Score:
        lat = ctx.latency_ms
        value = 1.0 if lat <= self.max_ms else self.max_ms / lat
        return self._s(value, lat <= self.max_ms, latency_ms=round(lat, 2), max_ms=self.max_ms)


# Цены по умолчанию — условные ($ за 1K токенов), задайте свои для реальной модели.
DEFAULT_PRICE_IN = 0.00005
DEFAULT_PRICE_OUT = 0.0002


def estimate_cost(
    usage: dict[str, int], price_in: float = DEFAULT_PRICE_IN, price_out: float = DEFAULT_PRICE_OUT
) -> float:
    return (
        usage.get("prompt_tokens", 0) / 1000 * price_in
        + usage.get("completion_tokens", 0) / 1000 * price_out
    )


class Cost(Scorer):
    """Стоимость кейса (из `ctx.cost_usd` или оценкой по токенам) не больше бюджета `max_usd`."""

    name = "cost"

    def __init__(
        self,
        max_usd: float = 0.01,
        price_in_per_1k: float = DEFAULT_PRICE_IN,
        price_out_per_1k: float = DEFAULT_PRICE_OUT,
    ) -> None:
        self.max_usd = float(max_usd)
        self.price_in = price_in_per_1k
        self.price_out = price_out_per_1k

    def score(self, case: Case, output: Any, ctx: ScoreContext) -> Score:
        cost = ctx.cost_usd
        if cost is None:
            cost = estimate_cost(ctx.usage, self.price_in, self.price_out)
        ok = cost <= self.max_usd
        value = 1.0 if ok else self.max_usd / cost
        return self._s(value, ok, cost_usd=cost, max_usd=self.max_usd, tokens=dict(ctx.usage))


# ---------------------------------------------------------------- registry
SCORERS: dict[str, type[Scorer]] = {
    "exact_match": ExactMatch,
    "contains": Contains,
    "regex": Regex,
    "json_schema": JsonSchema,
    "json_match": JsonMatch,
    "numeric_tolerance": NumericTolerance,
    "llm_judge": LLMJudge,
    "latency": Latency,
    "cost": Cost,
}

# Какой параметр конструктора получает позиционный аргумент из спецификации `name:arg`.
_PRIMARY_ARG = {
    "numeric_tolerance": ("tol", float),
    "llm_judge": ("threshold", int),
    "latency": ("max_ms", float),
    "cost": ("max_usd", float),
    "regex": ("pattern", str),
}

ScorerFn = Callable[..., Any]


def build_scorer(spec: str, llm: LLM | None = None) -> Scorer:
    """`"contains"`, `"latency:2000"`, `"numeric_tolerance:0.01"`, `"llm_judge:3"` → Scorer."""
    name, _, arg = spec.strip().partition(":")
    if name not in SCORERS:
        raise ValueError(f"Неизвестный скорер '{name}'. Доступны: {', '.join(SCORERS)}")
    kwargs: dict[str, Any] = {}
    if arg:
        if name not in _PRIMARY_ARG:
            raise ValueError(f"Скорер '{name}' не принимает аргумент")
        key, conv = _PRIMARY_ARG[name]
        kwargs[key] = conv(arg)
    if name == "llm_judge" and llm is not None:
        kwargs["llm"] = llm
    return SCORERS[name](**kwargs)


def build_scorers(specs: str | list[str], llm: LLM | None = None) -> list[Scorer]:
    items = specs.split(",") if isinstance(specs, str) else specs
    return [build_scorer(s, llm=llm) for s in items if s.strip()]


def scorer_name(scorer: ScorerFn) -> str:
    return getattr(scorer, "name", None) or getattr(scorer, "__name__", type(scorer).__name__)


def apply_scorer(scorer: ScorerFn, case: Case, output: Any, ctx: ScoreContext) -> Score:
    """Вызывает скорер любой формы и приводит результат к `Score`. Исключение скорера
    не валит прогон — превращается в провал с текстом ошибки."""
    name = scorer_name(scorer)
    try:
        target = scorer.score if isinstance(scorer, Scorer) else scorer
        n_params = len(inspect.signature(target).parameters)
        result = target(case, output, ctx) if n_params >= 3 else target(case, output)
    except Exception as e:  # noqa: BLE001 — ошибка скорера = провал кейса, а не падение прогона
        return Score(
            name=name, value=0.0, passed=False, details={"error": f"{type(e).__name__}: {e}"}
        )
    if isinstance(result, Score):
        if result.name != name and not isinstance(scorer, Scorer):
            result = result.model_copy(update={"name": name})
        return result
    if isinstance(result, bool):
        return Score(name=name, value=float(result), passed=result)
    if isinstance(result, int | float):
        v = max(0.0, min(1.0, float(result)))
        return Score(name=name, value=v, passed=v >= 0.5)
    return Score(
        name=name,
        value=0.0,
        passed=False,
        details={"error": f"скорер вернул {type(result).__name__}; ожидается Score|bool|float"},
    )
