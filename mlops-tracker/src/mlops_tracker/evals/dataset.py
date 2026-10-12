"""Датасеты для оценки: кейсы `{id, input, expected, metadata}` из JSONL / JSON / YAML."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class Case(BaseModel):
    id: str
    input: Any
    expected: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", mode="before")
    @classmethod
    def _id_to_str(cls, v: Any) -> str:
        return str(v)


class Dataset(BaseModel):
    name: str
    cases: list[Case]
    path: str | None = None

    @property
    def hash(self) -> str:
        """Контрольная сумма содержимого — для воспроизводимости (меняется при любой правке)."""
        payload = json.dumps(
            [c.model_dump() for c in self.cases], ensure_ascii=False, sort_keys=True, default=str
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):  # type: ignore[override]
        return iter(self.cases)

    @classmethod
    def from_records(
        cls,
        records: list[dict[str, Any]],
        name: str = "inline",
        default_metadata: dict[str, Any] | None = None,
        path: str | None = None,
    ) -> Dataset:
        cases: list[Case] = []
        seen: set[str] = set()
        for i, rec in enumerate(records):
            if not isinstance(rec, dict) or "input" not in rec:
                raise ValueError(f"Кейс #{i} должен быть объектом с полем 'input': {rec!r}")
            data = dict(rec)
            data.setdefault("id", f"case-{i + 1}")
            data["metadata"] = {**(default_metadata or {}), **(data.get("metadata") or {})}
            case = Case.model_validate(data)
            if case.id in seen:
                raise ValueError(f"Повторяющийся id кейса: {case.id}")
            seen.add(case.id)
            cases.append(case)
        return cls(name=name, cases=cases, path=path)


def load_dataset(path: str | Path) -> Dataset:
    """Загрузка датасета.

    * `.jsonl` — по кейсу на строку (пустые строки и строки с `#` пропускаются);
    * `.json` — список кейсов или `{"name", "metadata", "cases"}`;
    * `.yaml/.yml` — то же, что JSON. Поле верхнего уровня `metadata` подмешивается в каждый
      кейс (удобно задать одну JSON-схему на весь датасет).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    text = p.read_text(encoding="utf-8")
    name = p.stem
    default_meta: dict[str, Any] = {}
    if suffix == ".jsonl":
        records = []
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{lineno}: невалидный JSON: {e}") from e
    elif suffix in (".json", ".yaml", ".yml"):
        raw = json.loads(text) if suffix == ".json" else yaml.safe_load(text)
        if isinstance(raw, dict):
            name = raw.get("name", name)
            default_meta = raw.get("metadata") or {}
            records = raw.get("cases") or []
        elif isinstance(raw, list):
            records = raw
        else:
            raise ValueError(f"{p}: ожидается список кейсов или объект с ключом 'cases'")
    else:
        raise ValueError(f"Неподдерживаемый формат датасета: {suffix}")
    return Dataset.from_records(records, name=name, default_metadata=default_meta, path=str(p))
