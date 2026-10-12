"""Pydantic-схемы проекта."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SECTION_NAMES: tuple[str, ...] = ("contacts", "summary", "skills", "experience", "education")
Platform = Literal["linkedin", "indeed", "hh", "generic"]
PLATFORMS: tuple[str, ...] = ("linkedin", "indeed", "hh", "generic")


class DocumentInfo(BaseModel):
    """Результат извлечения текста из файла + признаки форматирования для ATS."""

    text: str
    fmt: str = "txt"
    tables: int = 0
    images: int = 0
    pages: int | None = None


class Contacts(BaseModel):
    name: str | None = None
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)


class Resume(BaseModel):
    raw_text: str
    sections: dict[str, str] = Field(default_factory=dict)
    contacts: Contacts = Field(default_factory=Contacts)
    language: Literal["ru", "en"] = "ru"
    doc: DocumentInfo | None = None

    def section(self, name: str) -> str:
        return self.sections.get(name, "")


class Vacancy(BaseModel):
    """Структурированная вакансия. Эта же схема отдаётся LLM как JSON Schema."""

    title: str = Field("", description="Название позиции")
    must_have: list[str] = Field(default_factory=list, description="Обязательные навыки")
    nice_to_have: list[str] = Field(default_factory=list, description="Навыки «будет плюсом»")
    min_years: float | None = Field(None, description="Минимальный требуемый опыт, лет")
    keywords: list[str] = Field(default_factory=list, description="Все значимые ключевые слова")
    alternatives: list[list[str]] = Field(
        default_factory=list,
        description="Взаимозаменяемые навыки («LangChain или LlamaIndex»): достаточно любого",
    )
    language: Literal["ru", "en"] = "ru"
    raw_text: str = Field("", exclude=True)


class ScoreComponent(BaseModel):
    name: str
    weight: float
    score: float = Field(..., ge=0.0, le=1.0, description="Нормированный балл 0..1")
    points: float = Field(..., description="Вклад в итог: weight * score")
    details: str = ""


class ATSScore(BaseModel):
    total: float = Field(..., ge=0.0, le=100.0)
    components: list[ScoreComponent]
    matched_keywords: list[str] = Field(default_factory=list)
    missing_must_have: list[str] = Field(default_factory=list)
    missing_nice_to_have: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    resume_years: float | None = None
    required_years: float | None = None
    suggestions: list[str] = Field(default_factory=list)

    @property
    def missing_keywords(self) -> list[str]:
        return self.missing_must_have + self.missing_nice_to_have

    def component(self, name: str) -> ScoreComponent:
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(name)


class ChangeEntry(BaseModel):
    section: str
    action: Literal[
        "rephrased", "reordered", "keywords_added", "unchanged", "reverted", "formatted"
    ]
    detail: str = ""


class LLMTailorOutput(BaseModel):
    """Схема ответа LLM при адаптации резюме."""

    sections: dict[str, str]
    changes: list[ChangeEntry] = Field(default_factory=list)


class TailoredResume(BaseModel):
    platform: str
    vacancy_title: str
    language: Literal["ru", "en"] = "ru"
    sections: dict[str, str]
    changelog: list[ChangeEntry] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    score_before: float | None = None
    score_after: float | None = None


class CoverLetter(BaseModel):
    platform: str
    vacancy_title: str
    text: str
    used_keywords: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
