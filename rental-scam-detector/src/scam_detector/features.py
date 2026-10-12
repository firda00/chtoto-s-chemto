"""Сборка вектора признаков объявления: цена vs район, дубликаты фото, текст, корпусные сигналы."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .images import PhotoDupStats, PhotoHash, PhotoIndex, dup_stats, hash_image
from .models import Listing, PhotoMatch
from .text_features import (
    DistrictTable,
    PhoneRegistry,
    PriceDeviation,
    SignalHit,
    TextIndex,
    card_number,
    generic_description,
    normalize_district,
    off_platform_contact,
    owner_abroad,
    prepayment,
    price_deviation,
    to_usd,
    urgency,
)

NUMERIC_FEATURES = [
    "price_usd_m2",
    "price_ratio_to_median",
    "price_z",
    "too_good_price",
    "district_known",
    "area_m2",
    "rooms",
    "n_photos",
    "photo_dup_count",
    "photo_dup_listings",
    "photo_dup_diff_district",
    "photo_dup_price_diff",
    "urgency_hits",
    "prepayment_hits",
    "owner_abroad_hits",
    "offplatform_hits",
    "card_number",
    "generic_score",
    "desc_words",
    "text_dup_jaccard",
    "phone_reuse",
]
CAT_FEATURES = ["district"]
TEXT_DUP_MIN_JACCARD = 0.5
FEATURE_NAMES = NUMERIC_FEATURES + CAT_FEATURES


@dataclass
class Evidence:
    """Сырые находки — из них строятся человекочитаемые причины."""

    price: PriceDeviation
    urgency: SignalHit
    prepayment: SignalHit
    owner_abroad: SignalHit
    off_platform: SignalHit
    card: SignalHit
    generic_phrases: SignalHit
    photo: PhotoDupStats
    photo_matches: list[PhotoMatch] = field(default_factory=list)
    text_dup_with: str | None = None
    phone_reuse: int = 0


@dataclass
class ListingFeatures:
    values: dict[str, float | str]
    evidence: Evidence


class FeatureContext:
    """Справочник цен + корпус известных объявлений (фото-индекс, тексты, телефоны)."""

    def __init__(
        self,
        districts: DistrictTable,
        photo_index: PhotoIndex | None = None,
        text_index: TextIndex | None = None,
        phones: PhoneRegistry | None = None,
    ) -> None:
        self.districts = districts
        self.photo_index = photo_index or PhotoIndex()
        self.text_index = text_index or TextIndex()
        self.phones = phones or PhoneRegistry()

    @classmethod
    def from_listings(
        cls,
        listings: list[Listing],
        districts: DistrictTable | None = None,
        photo_hashes: dict[str, list[PhotoHash]] | None = None,
    ) -> FeatureContext:
        """Строит контекст по корпусу. `photo_hashes` — готовые хэши (синтетика, без файлов)."""
        districts = districts or DistrictTable.load()
        ctx = cls(districts)
        for lst in listings:
            usd = to_usd(lst.price, lst.currency)
            if photo_hashes is not None and lst.id in photo_hashes:
                for i, h in enumerate(photo_hashes[lst.id]):
                    ctx.photo_index.add(lst.id, f"photo_{i}", h, usd, lst.district)
            else:
                ctx.photo_index.add_photos(lst.id, lst.photos, usd, lst.district)
            ctx.phones.add(lst.id, lst.contact_phone)
        ctx.text_index.fit([x.id for x in listings], [x.text for x in listings])
        return ctx

    def _photo_hashes(self, listing: Listing) -> list[tuple[str, PhotoHash]]:
        indexed = self.photo_index.hashes_for(listing.id)
        if indexed is not None:
            return indexed
        out = []
        for p in listing.photos:
            try:
                out.append((Path(p).name, hash_image(p)))
            except (OSError, ValueError):
                continue
        return out

    def features(self, listing: Listing) -> ListingFeatures:
        text = listing.text
        price = price_deviation(
            listing.price, listing.currency, listing.area_m2, listing.district, self.districts
        )
        district_key = self.districts.canonical(listing.district)
        hashes = self._photo_hashes(listing)
        matches = self.photo_index.query(
            [h for _, h in hashes], exclude_listing=listing.id, photo_names=[n for n, _ in hashes]
        )

        def norm(d: str) -> str:
            return self.districts.canonical(d) or normalize_district(d)

        photo = dup_stats(matches, price.price_usd, norm(listing.district), norm)
        u, pp, ab, op, card = (
            urgency(text),
            prepayment(text),
            owner_abroad(text),
            off_platform_contact(text),
            card_number(text),
        )
        generic, generic_phrases = generic_description(listing.description)
        jac, dup_with = self.text_index.max_jaccard(text, exclude_id=listing.id)
        if jac < TEXT_DUP_MIN_JACCARD:
            # Ниже порога — это общая «шаблонная» лексика объявлений, а не копия текста.
            # Обнуляем, чтобы признак не зависел от размера корпуса.
            jac, dup_with = 0.0, None
        phone_reuse = self.phones.reuse_count(listing.contact_phone, exclude_id=listing.id)

        values: dict[str, float | str] = {
            "price_usd_m2": round(price.usd_per_m2, 3),
            "price_ratio_to_median": round(price.ratio, 4),
            "price_z": round(price.z, 4),
            "too_good_price": float(price.too_good),
            "district_known": float(price.district_known),
            "area_m2": float(listing.area_m2),
            "rooms": float(listing.rooms),
            "n_photos": float(len(hashes)),
            "photo_dup_count": float(photo.dup_photos),
            "photo_dup_listings": float(photo.dup_listings),
            "photo_dup_diff_district": float(photo.diff_district),
            "photo_dup_price_diff": round(photo.max_price_diff, 4),
            "urgency_hits": float(u.count),
            "prepayment_hits": float(pp.count),
            "owner_abroad_hits": float(ab.count),
            "offplatform_hits": float(op.count),
            "card_number": float(card.count > 0),
            "generic_score": round(generic, 4),
            "desc_words": float(len(re.findall(r"\w+", listing.description))),
            "text_dup_jaccard": round(jac, 4),
            "phone_reuse": float(phone_reuse),
            "district": district_key or "unknown",
        }
        evidence = Evidence(
            price=price,
            urgency=u,
            prepayment=pp,
            owner_abroad=ab,
            off_platform=op,
            card=card,
            generic_phrases=generic_phrases,
            photo=photo,
            photo_matches=matches,
            text_dup_with=dup_with,
            phone_reuse=phone_reuse,
        )
        return ListingFeatures(values, evidence)


# --------------------------------------------------------------------------- загрузка объявлений


def load_listing(path: str | Path) -> Listing:
    """Читает JSON объявления; относительные пути фото — относительно файла."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["photos"] = [
        str(p if Path(p).is_absolute() else (path.parent / p).resolve())
        for p in data.get("photos", [])
    ]
    return Listing.model_validate(data)


def load_corpus(directory: str | Path) -> list[Listing]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return [load_listing(p) for p in sorted(directory.glob("*.json"))]
