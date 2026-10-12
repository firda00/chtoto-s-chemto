"""Pydantic-схемы: объявление и отчёт о риске."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class Listing(BaseModel):
    """Объявление об аренде (цена — за месяц)."""

    id: str
    title: str
    description: str
    price: float = Field(gt=0)
    currency: str = "USD"
    area_m2: float = Field(gt=0)
    rooms: int = Field(ge=0)
    district: str
    contact_phone: str = ""
    contact_name: str = ""
    photos: list[Path] = Field(default_factory=list)
    posted_at: datetime | None = None
    source_url: str | None = None

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.description}"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float) -> RiskLevel:
        if score < 25:
            return cls.LOW
        if score < 50:
            return cls.MEDIUM
        if score < 75:
            return cls.HIGH
        return cls.CRITICAL


class FeatureContribution(BaseModel):
    """SHAP-вклад признака в log-odds (положительный — в сторону «мошенничество»)."""

    feature: str
    value: float | str
    contribution: float


class PhotoMatch(BaseModel):
    """Найденный почти-дубликат фото в другом объявлении."""

    photo: str
    other_listing_id: str
    other_photo: str
    phash_distance: int
    dhash_distance: int
    other_price_usd: float | None = None
    other_district: str | None = None

    @property
    def distance(self) -> int:
        return max(self.phash_distance, self.dhash_distance)


class ScamReport(BaseModel):
    listing_id: str
    score: float = Field(ge=0, le=100, description="Scam Score 0–100 (= P(scam) × 100)")
    risk_level: RiskLevel
    base_value: float = Field(description="Ожидаемое значение модели (log-odds) до учёта признаков")
    contributions: list[FeatureContribution]
    reasons: list[str]
    duplicate_photos: list[PhotoMatch] = Field(default_factory=list)
