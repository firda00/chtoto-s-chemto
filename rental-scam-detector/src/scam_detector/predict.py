"""Скоринг объявления: CatBoost → Scam Score, SHAP-вклады признаков, причины по-русски."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from .config import model_path as default_model_path
from .features import FEATURE_NAMES, Evidence, FeatureContext, ListingFeatures
from .models import FeatureContribution, Listing, RiskLevel, ScamReport
from .train import make_pool

MIN_REASON_SHAP = 0.02  # причина показывается, если сигнал активен и толкает к «мошенничеству»


def _quote(fragments: list[str], k: int = 3) -> str:
    return ", ".join(f"«{f}»" for f in fragments[:k])


def _reason(feature: str, values: dict, ev: Evidence) -> str | None:
    """Человекочитаемое объяснение признака или None, если сигнал не активен."""
    p = ev.price
    match feature:
        case "price_ratio_to_median" | "price_z" | "too_good_price" | "price_usd_m2":
            if p.ratio < 0.8:
                return (
                    f"Цена {p.usd_per_m2:.1f} $/м² на {(1 - p.ratio):.0%} ниже медианы района "
                    f"{p.district_name} ({p.median_usd_m2:.1f} $/м²)"
                )
        case (
            "photo_dup_count"
            | "photo_dup_listings"
            | "photo_dup_diff_district"
            | ("photo_dup_price_diff")
        ):
            if ev.photo_matches:
                m = ev.photo_matches[0]
                extra = []
                if ev.photo.diff_district:
                    extra.append(f"другой район: {m.other_district}")
                if ev.photo.max_price_diff >= 0.15:
                    extra.append(f"цена отличается на {ev.photo.max_price_diff:.0%}")
                tail = f" ({'; '.join(extra)})" if extra else ""
                return (
                    f"{ev.photo.dup_photos} фото совпадают с объявлением {m.other_listing_id}"
                    f"{tail}, расстояние хэшей {m.distance} бит"
                )
        case "urgency_hits":
            if ev.urgency.count:
                return f"Давление срочностью: {_quote(ev.urgency.fragments)}"
        case "prepayment_hits":
            if ev.prepayment.count:
                return (
                    f"Просьба о предоплате/переводе до просмотра: {_quote(ev.prepayment.fragments)}"
                )
        case "card_number":
            if ev.card.count:
                return "В тексте указан номер банковской карты"
        case "owner_abroad_hits":
            if ev.owner_abroad.count:
                return (
                    f"Легенда «хозяин далеко, покажет курьер»: {_quote(ev.owner_abroad.fragments)}"
                )
        case "offplatform_hits":
            if ev.off_platform.count:
                return f"Контакт только вне площадки: {_quote(ev.off_platform.fragments)}"
        case "generic_score" | "desc_words":
            if values["generic_score"] >= 0.66:
                phrases = (
                    f": {_quote(ev.generic_phrases.fragments)}" if ev.generic_phrases.count else ""
                )
                return (
                    f"Шаблонное описание без конкретики ({int(values['desc_words'])} слов){phrases}"
                )
        case "text_dup_jaccard":
            if values["text_dup_jaccard"] > 0 and ev.text_dup_with:
                return (
                    f"Текст на {values['text_dup_jaccard']:.0%} совпадает с объявлением "
                    f"{ev.text_dup_with}"
                )
        case "phone_reuse":
            if ev.phone_reuse >= 1:
                return f"Тот же телефон ещё в других объявлениях: {ev.phone_reuse}"
        case "district_known":
            if not p.district_known:
                return "Район не распознан — цена сравнивается со средней по городу"
    return None


class ScamDetector:
    def __init__(self, model: CatBoostClassifier) -> None:
        self.model = model

    @classmethod
    def load(cls, path: str | Path | None = None) -> ScamDetector:
        path = Path(path or default_model_path())
        if not path.exists():
            raise FileNotFoundError(
                f"Модель не найдена: {path}. Сначала: uv run scam make-dataset && uv run scam train"
            )
        model = CatBoostClassifier()
        model.load_model(str(path))
        return cls(model)

    def report(self, listing: Listing, ctx: FeatureContext) -> ScamReport:
        return self.report_from_features(listing.id, ctx.features(listing))

    def report_from_features(self, listing_id: str, lf: ListingFeatures) -> ScamReport:
        df = pd.DataFrame([lf.values])
        pool = make_pool(df)
        proba = float(self.model.predict_proba(pool)[0, 1])
        shap = self.model.get_feature_importance(data=pool, type="ShapValues")[0]
        contrib_values = shap[:-1]
        base_value = float(shap[-1])

        contributions = [
            FeatureContribution(
                feature=name, value=lf.values[name], contribution=round(float(c), 4)
            )
            for name, c in zip(FEATURE_NAMES, contrib_values, strict=True)
        ]
        contributions.sort(key=lambda c: -abs(c.contribution))

        reasons: list[str] = []
        for idx in np.argsort(-contrib_values):
            if contrib_values[idx] < MIN_REASON_SHAP:
                break
            text = _reason(FEATURE_NAMES[idx], lf.values, lf.evidence)
            if text and text not in reasons:
                reasons.append(text)
        if not reasons:
            reasons.append("Явных признаков мошенничества не найдено")

        score = round(proba * 100, 1)
        return ScamReport(
            listing_id=listing_id,
            score=score,
            risk_level=RiskLevel.from_score(score),
            base_value=round(base_value, 4),
            contributions=contributions,
            reasons=reasons,
            duplicate_photos=lf.evidence.photo_matches,
        )


def build_context(
    corpus_dir: str | Path | None, extra: list[Listing] | None = None, districts=None
) -> FeatureContext:
    """Контекст = справочник цен + корпус объявлений из папки (+ проверяемые, если их там нет)."""
    from .features import load_corpus
    from .text_features import DistrictTable

    corpus = load_corpus(corpus_dir) if corpus_dir else []
    known = {x.id for x in corpus}
    corpus += [x for x in (extra or []) if x.id not in known]
    return FeatureContext.from_listings(corpus, districts or DistrictTable.load())
