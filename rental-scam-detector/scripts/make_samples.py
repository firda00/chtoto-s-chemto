"""Генерирует демо-объявления data/samples/*.json и крошечные синтетические фото к ним.

listing_02 использует почти-дубликат фото из listing_01 (ресайз + JPEG + яркость) —
классический паттерн «те же фото, другой район и цена».
Запуск: uv run python scripts/make_samples.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scam_detector.config import DATA_DIR
from scam_detector.synth_images import near_duplicate, render_room

# имя файла → (сид композиции, перезалитая копия?)
IMAGES = {
    "flat_a_1.png": (11, False),
    "flat_a_2.png": (12, False),
    "flat_a_1_repost.jpg": (11, True),
    "flat_b_1.png": (21, False),
    "flat_b_2.png": (22, False),
    "flat_c_1.png": (41, False),
    "flat_d_1.png": (51, False),
    "flat_d_2.png": (52, False),
    "flat_e_1.png": (61, False),
}

LISTINGS = [
    {
        "id": "listing_01",
        "title": "2-комн. квартира, Юнусабад, 62 м²",
        "description": (
            "Сдаётся 2-комнатная квартира в Юнусабаде, 62 м², 4/9 этаж. Рядом метро Минор, "
            "ориентир — Мега Планет. Евроремонт 2023 года, кухня 10 м². Есть: кондиционер, "
            "стиральная машина, холодильник, Wi-Fi. Только на длительный срок, договор аренды, "
            "залог за один месяц при подписании. Показ в любое время по договорённости."
        ),
        "price": 460,
        "currency": "USD",
        "area_m2": 62,
        "rooms": 2,
        "district": "Юнусабад",
        "contact_phone": "+998 97 455 12 80",
        "contact_name": "Дилноза",
        "photos": ["images/flat_a_1.png", "images/flat_a_2.png"],
        "posted_at": "2026-09-20T10:15:00",
        "source_url": "https://example.local/rent/listing_01",
    },
    {
        "id": "listing_02",
        "title": "СРОЧНО 2-комн. квартира, Мирабад",
        "description": (
            "СРОЧНО! Отличная квартира, всё есть. Я сейчас нахожусь за границей, ключи передам "
            "через курьера. Для брони нужна предоплата 50$ на карту 8600 1234 5678 9012. "
            "Звонки не принимаю, пишите только в Telegram @arenda_tash."
        ),
        "price": 250,
        "currency": "USD",
        "area_m2": 60,
        "rooms": 2,
        "district": "Мирабад",
        "contact_phone": "+998 90 111 22 33",
        "contact_name": "Хозяин",
        "photos": ["images/flat_a_1_repost.jpg"],
        "posted_at": "2026-10-01T22:40:00",
        "source_url": "https://example.local/rent/listing_02",
    },
    {
        "id": "listing_03",
        "title": "1-комн. квартира, Чиланзар 9 квартал",
        "description": (
            "Сдаю 1-комнатную квартиру, Чиланзар, 40 м², 2/4 этаж. До метро Чиланзар 7 минут "
            "пешком. Свежий косметический ремонт, кухня 8 м². Есть газовая плита, бойлер, "
            "холодильник. Семье или работающим, без животных. Коммунальные платежи отдельно. "
            "Звоните после 18:00."
        ),
        "price": 3300000,
        "currency": "UZS",
        "area_m2": 40,
        "rooms": 1,
        "district": "Чиланзар",
        "contact_phone": "+998 93 270 44 19",
        "contact_name": "Азиз",
        "photos": ["images/flat_b_1.png", "images/flat_b_2.png"],
        "posted_at": "2026-09-28T09:00:00",
        "source_url": "https://example.local/rent/listing_03",
    },
    {
        "id": "listing_04",
        "title": "Shoshilinch! 3 xonali kvartira, Yakkasaroy",
        "description": (
            "Shoshilinch! Faqat bugun shu narxda. Zo'r kvartira, hammasi bor. Hozir chet eldaman, "
            "kalitni pochta orqali yuboraman. Oldindan to'lov kartaga o'tkazing. "
            "Faqat telegramga yozing."
        ),
        "price": 3800000,
        "currency": "UZS",
        "area_m2": 85,
        "rooms": 3,
        "district": "Yakkasaroy",
        "contact_phone": "+998 (90) 111-22-33",
        "contact_name": "Jasur",
        "photos": ["images/flat_c_1.png"],
        "posted_at": "2026-10-02T23:10:00",
        "source_url": "https://example.local/rent/listing_04",
    },
    {
        "id": "listing_05",
        "title": "3-bedroom apartment, Mirzo Ulugbek",
        "description": (
            "Bright 3-bedroom apartment in Mirzo Ulugbek, 95 m², floor 6 of 9. Near Buyuk Ipak "
            "Yo'li metro station, landmark: Korzinka. Renovated in 2024, fully furnished, kitchen "
            "14 m². Includes: air conditioning, dishwasher, washing machine, parking. Long-term "
            "only, contract available. Utilities paid separately. Viewing any day by appointment."
        ),
        "price": 780,
        "currency": "USD",
        "area_m2": 95,
        "rooms": 3,
        "district": "Mirzo Ulug'bek",
        "contact_phone": "+998 99 812 35 07",
        "contact_name": "Malika",
        "photos": ["images/flat_d_1.png", "images/flat_d_2.png"],
        "posted_at": "2026-09-25T14:30:00",
        "source_url": "https://example.local/rent/listing_05",
    },
    {
        "id": "listing_06",
        "title": "Квартира в Сергели",
        "description": "Хорошая квартира, всё есть. Недорого. Пишите в WhatsApp.",
        "price": 125,
        "currency": "USD",
        "area_m2": 38,
        "rooms": 1,
        "district": "Сергели",
        "contact_phone": "+998 94 620 71 55",
        "contact_name": "Ольга",
        "photos": ["images/flat_e_1.png"],
        "posted_at": "2026-10-03T08:05:00",
        "source_url": "https://example.local/rent/listing_06",
    },
]


def make_samples(out_dir: Path) -> list[Path]:
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for name, (seed, repost) in IMAGES.items():
        img = render_room(seed)
        if repost:
            img = near_duplicate(img, seed)
        img.save(img_dir / name, **({"quality": 85} if name.endswith(".jpg") else {}))
    paths = []
    for item in LISTINGS:
        p = out_dir / f"{item['id']}.json"
        p.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "samples")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    paths = make_samples(args.out)
    print(f"Записано {len(paths)} объявлений и {len(IMAGES)} фото в {args.out}")


if __name__ == "__main__":
    main()
