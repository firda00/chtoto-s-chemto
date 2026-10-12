"""Перцептивное хэширование фото и поиск почти-дубликатов между объявлениями."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image

from .config import PHOTO_HAMMING_THRESHOLD
from .models import PhotoMatch

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhotoHash:
    """64-битные pHash и dHash одного фото (как целые числа)."""

    phash: int
    dhash: int

    @classmethod
    def from_hex(cls, phash_hex: str, dhash_hex: str) -> PhotoHash:
        return cls(int(phash_hex, 16), int(dhash_hex, 16))

    def distance(self, other: PhotoHash) -> tuple[int, int]:
        return (self.phash ^ other.phash).bit_count(), (self.dhash ^ other.dhash).bit_count()


def _to_int(h: imagehash.ImageHash) -> int:
    return int(str(h), 16)


def hash_image(image: Image.Image | str | Path) -> PhotoHash:
    """pHash + dHash (8×8 = 64 бита каждый)."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as img:
            return hash_image(img.copy())
    img = image.convert("RGB")
    return PhotoHash(_to_int(imagehash.phash(img)), _to_int(imagehash.dhash(img)))


@dataclass
class _Entry:
    listing_id: str
    photo: str
    price_usd: float | None
    district: str | None


class PhotoIndex:
    """Индекс хэшей фото всех известных объявлений.

    Поиск векторизован через numpy (xor + popcount), так что индекс на десятки тысяч фото
    обрабатывается за миллисекунды на запрос.
    """

    def __init__(self, threshold: int = PHOTO_HAMMING_THRESHOLD) -> None:
        self.threshold = threshold
        self._entries: list[_Entry] = []
        self._phash: list[int] = []
        self._dhash: list[int] = []
        self._by_listing: dict[str, list[tuple[str, PhotoHash]]] = {}
        self._arrays: tuple[np.ndarray, np.ndarray] | None = None

    def __len__(self) -> int:
        return len(self._entries)

    def add(
        self,
        listing_id: str,
        photo: str,
        h: PhotoHash,
        price_usd: float | None = None,
        district: str | None = None,
    ) -> None:
        self._entries.append(_Entry(listing_id, photo, price_usd, district))
        self._phash.append(h.phash)
        self._dhash.append(h.dhash)
        self._by_listing.setdefault(listing_id, []).append((photo, h))
        self._arrays = None

    def add_photos(
        self,
        listing_id: str,
        photos: list[Path],
        price_usd: float | None = None,
        district: str | None = None,
    ) -> list[PhotoHash]:
        """Хэширует файлы и добавляет в индекс; отсутствующие/битые файлы пропускаются."""
        hashes = []
        for p in photos:
            try:
                h = hash_image(p)
            except (OSError, ValueError) as exc:
                log.warning("Не удалось прочитать фото %s: %s", p, exc)
                continue
            self.add(listing_id, Path(p).name, h, price_usd, district)
            hashes.append(h)
        return hashes

    def hashes_for(self, listing_id: str) -> list[tuple[str, PhotoHash]] | None:
        """(имя фото, хэш) для уже проиндексированного объявления."""
        return self._by_listing.get(listing_id)

    def _get_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if self._arrays is None:
            self._arrays = (
                np.array(self._phash, dtype=np.uint64),
                np.array(self._dhash, dtype=np.uint64),
            )
        return self._arrays

    def query(
        self,
        hashes: list[PhotoHash],
        exclude_listing: str | None = None,
        photo_names: list[str] | None = None,
    ) -> list[PhotoMatch]:
        """Все пары (наше фото, чужое фото) с max(dist_pHash, dist_dHash) <= threshold."""
        if not self._entries or not hashes:
            return []
        ph_arr, dh_arr = self._get_arrays()
        names = photo_names or [f"photo_{i}" for i in range(len(hashes))]
        matches: list[PhotoMatch] = []
        for name, h in zip(names, hashes, strict=False):
            dp = np.bitwise_count(ph_arr ^ np.uint64(h.phash))
            dd = np.bitwise_count(dh_arr ^ np.uint64(h.dhash))
            idx = np.nonzero(np.maximum(dp, dd) <= self.threshold)[0]
            for i in idx:
                e = self._entries[int(i)]
                if e.listing_id == exclude_listing:
                    continue
                matches.append(
                    PhotoMatch(
                        photo=name,
                        other_listing_id=e.listing_id,
                        other_photo=e.photo,
                        phash_distance=int(dp[i]),
                        dhash_distance=int(dd[i]),
                        other_price_usd=e.price_usd,
                        other_district=e.district,
                    )
                )
        matches.sort(key=lambda m: (m.distance, m.other_listing_id))
        return matches


@dataclass
class PhotoDupStats:
    """Сводка по дубликатам фото для признаков модели."""

    dup_photos: int = 0  # сколько наших фото нашлось в чужих объявлениях
    dup_listings: int = 0  # в скольких разных объявлениях
    diff_district: int = 0  # 1 — те же фото в другом районе
    max_price_diff: float = 0.0  # max |Δцена| / max(цена) среди совпавших объявлений


def dup_stats(
    matches: list[PhotoMatch],
    price_usd: float,
    district_key: str,
    normalize_district=lambda s: s,
) -> PhotoDupStats:
    """Паттерн «те же фото, другая цена/район»."""
    if not matches:
        return PhotoDupStats()
    others = {m.other_listing_id: m for m in matches}
    diff_district = any(
        m.other_district is not None and normalize_district(m.other_district) != district_key
        for m in others.values()
    )
    price_diffs = [
        abs(price_usd - m.other_price_usd) / max(price_usd, m.other_price_usd)
        for m in others.values()
        if m.other_price_usd
    ]
    return PhotoDupStats(
        dup_photos=len({m.photo for m in matches}),
        dup_listings=len(others),
        diff_district=int(diff_district),
        max_price_diff=max(price_diffs, default=0.0),
    )
