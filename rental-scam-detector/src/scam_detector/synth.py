"""Синтетический размеченный датасет объявлений аренды (Ташкент).

Генерируются не готовые признаки, а *объявления* (текст RU/UZ/EN, цена, район, телефон,
перцептивные хэши фото), которые затем проходят через тот же `FeatureContext`, что и боевые
объявления. Корреляции заданы вероятностями ниже: у мошенников чаще занижена цена, есть
предоплата/срочность/«хозяин за границей», украденные фото и текст, общие телефоны; у честных
объявлений эти сигналы тоже изредка встречаются (шум), а агентства переиспользуют телефоны и
перезаливают свои же объявления.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import uzs_per_usd
from .features import FEATURE_NAMES, FeatureContext
from .images import PhotoHash
from .models import Listing
from .text_features import DistrictTable

# Вероятности сигналов: (у мошенников, у честных)
P = {
    "urgency": (0.45, 0.12),
    "prepayment": (0.40, 0.06),
    "abroad": (0.30, 0.02),
    "offplatform": (0.45, 0.15),
    "card": (0.15, 0.005),
    "generic": (0.35, 0.12),
    "short_text": (0.45, 0.20),  # обрезанное описание (2–4 предложения)
    "copy_text": (0.30, 0.0),
    "stolen_photos": (0.50, 0.0),
    "shared_phone": (0.55, 0.20),  # пул телефонов мошенников / агентств
    "repost": (0.0, 0.05),  # агентство перезаливает своё же объявление
    "market_price": (0.25, 0.0),  # «умные» мошенники ставят почти рыночную цену
}
LABEL_NOISE = 0.02  # доля ошибок разметки (как у реальных модераторов)

METRO_RU = [
    "Минор",
    "Бодомзор",
    "Юнусабад",
    "Чиланзар",
    "Новза",
    "Космонавтлар",
    "Ойбек",
    "Буюк Ипак Йули",
    "Хамида Алимджана",
    "Пушкин",
    "Мустакиллик майдони",
    "Гафура Гуляма",
]
METRO_LAT = [
    "Minor",
    "Bodomzor",
    "Yunusobod",
    "Chilonzor",
    "Novza",
    "Kosmonavtlar",
    "Oybek",
    "Buyuk Ipak Yo'li",
    "Hamid Olimjon",
    "Pushkin",
    "Mustaqillik maydoni",
    "G'afur G'ulom",
]
STREETS = [
    "Амира Темура",
    "Мустакиллик",
    "Шота Руставели",
    "Бабура",
    "Фергана йули",
    "Нукус",
    "Мукими",
    "Катартал",
    "Бунёдкор",
    "Чимкент",
    "Осиё",
    "Махтумкули",
    "Лабзак",
    "Кичик халка",
]
LANDMARK_RU = [
    "Корзинка",
    "Мега Планет",
    "ТЦ Самарканд Дарвоза",
    "Анхор",
    "Ц-1",
    "Хилтон",
    "Next",
    "Национальный парк",
    "Алайский базар",
    "Ташкент Сити",
    "ТТЗ",
    "Компас",
]
LANDMARK_LAT = [
    "Korzinka",
    "Mega Planet",
    "Samarqand Darvoza",
    "Anhor",
    "C-1",
    "Hilton",
    "Next",
    "Milliy bog'",
    "Oloy bozori",
    "Tashkent City",
    "TTZ",
    "Kompas",
]
AMEN_RU = [
    "кондиционер",
    "стиральная машина",
    "холодильник",
    "Wi-Fi",
    "газовая плита",
    "бойлер",
    "парковка во дворе",
    "балкон",
    "лифт",
    "посудомоечная машина",
    "телевизор",
    "шкаф-купе",
]
AMEN_UZ = [
    "konditsioner",
    "kir yuvish mashinasi",
    "muzlatgich",
    "Wi-Fi",
    "gaz plita",
    "boyler",
    "avtoturargoh",
    "balkon",
    "lift",
    "televizor",
]
AMEN_EN = [
    "air conditioning",
    "washing machine",
    "fridge",
    "Wi-Fi",
    "gas stove",
    "water heater",
    "parking",
    "balcony",
    "elevator",
    "dishwasher",
    "smart TV",
]

LEGIT = {
    "ru": {
        "intro": (
            "Сдаётся {rooms}-комнатная квартира, {district_ru}, {area} м², {floor}/{floors} этаж."
        ),
        "details": [
            "Кухня {kitchen} м², потолки {ceil} м.",
            "Улица {street}, дом {house}.",
            "До остановки {minutes} минут.",
            "Дом {floors}-этажный, {year} года постройки.",
            "Окна во двор, кухня {kitchen} м².",
        ],
        "location": [
            "Рядом метро {metro}.",
            "Ориентир — {landmark}.",
            "В шаговой доступности школа, садик и {landmark}.",
            "До метро {metro} 5 минут пешком.",
        ],
        "condition": [
            "Евроремонт {year} года.",
            "Свежий косметический ремонт.",
            "Ремонт средний, мебель частично.",
            "Новая мебель и техника.",
            "Квартира после капремонта, тёплые полы.",
        ],
        "terms": [
            "Только на длительный срок.",
            "Семье или работающим.",
            "Без животных.",
            "Оплата помесячно, залог за один месяц при подписании договора.",
            "Коммунальные услуги оплачиваются отдельно.",
            "Договор аренды, прописка возможна.",
        ],
        "viewing": [
            "Показ в любое время по договорённости.",
            "Звоните после 18:00.",
            "Просмотр по выходным.",
            "Хозяйка живёт рядом, покажет в любой день.",
        ],
        "amen": "Есть: {amen}.",
        "title": [
            "{rooms}-комн. квартира, {district_ru}",
            "Аренда {rooms}к, {district_ru}, {area} м²",
            "Сдаю {rooms}-комнатную в районе {district_ru}",
        ],
    },
    "uz": {
        "intro": "{district} tumanida {rooms} xonali kvartira ijaraga beriladi, {area} m², "
        "{floor}/{floors} qavat.",
        "details": [
            "Oshxona {kitchen} m², shift {ceil} m.",
            "{street} ko'chasi, {house}-uy.",
            "Bekatgacha {minutes} daqiqa.",
            "Uy {year}-yilda qurilgan.",
        ],
        "location": [
            "Metro {metro} yaqinida.",
            "Mo'ljal: {landmark}.",
            "Yaqinida maktab, bog'cha va {landmark} bor.",
        ],
        "condition": [
            "Yevroremont qilingan ({year}).",
            "Yangi ta'mir.",
            "Mebel va texnika bor.",
            "O'rtacha ta'mir, mebel qisman.",
        ],
        "terms": [
            "Uzoq muddatga.",
            "Oilaga beriladi.",
            "Shartnoma asosida.",
            "Kommunal to'lovlar alohida.",
            "Uy hayvonlarisiz.",
        ],
        "viewing": [
            "Ko'rish istalgan vaqtda.",
            "Kechqurun qo'ng'iroq qiling.",
            "Dam olish kunlari ko'rsataman.",
        ],
        "amen": "Bor: {amen}.",
        "title": ["{rooms} xonali kvartira, {district}", "Ijaraga {rooms} xonali, {district}"],
    },
    "en": {
        "intro": "Bright {rooms}-bedroom apartment in {district}, {area} m², floor {floor} of "
        "{floors}.",
        "details": [
            "Kitchen {kitchen} m², ceilings {ceil} m.",
            "{street} street, building {house}.",
            "{minutes} minutes to the bus stop.",
            "Built in {year}.",
        ],
        "location": [
            "Near {metro} metro station.",
            "Landmark: {landmark}.",
            "Walking distance to {landmark} and a school.",
        ],
        "condition": [
            "Renovated in {year}, fully furnished.",
            "Fresh renovation.",
            "Modern furniture and appliances.",
            "Partially furnished.",
        ],
        "terms": [
            "Long-term only, contract available.",
            "Utilities paid separately.",
            "Suitable for a family or expats.",
            "No pets please.",
        ],
        "viewing": [
            "Viewing any day by appointment.",
            "Call after 6 pm.",
            "The owner lives nearby and can show it any day.",
        ],
        "amen": "Includes: {amen}.",
        "title": ["{rooms}-bedroom apartment, {district}", "Apartment for rent in {district}"],
    },
}

SCAM = {
    "ru": {
        "urgency": [
            "СРОЧНО!",
            "Только сегодня такая цена!",
            "Успейте, желающих много!",
            "Кто первый внесёт бронь — тот и заселяется.",
        ],
        "prepayment": [
            "Для брони нужна предоплата 50$ на карту.",
            "Задаток до просмотра, иначе не держу.",
            "Переведите залог, и я вышлю договор.",
            "Оплата брони сегодня, потом показ.",
        ],
        "abroad": [
            "Я сейчас нахожусь за границей, ключи передам через курьера.",
            "Хозяин живёт в Москве, показать не смогу.",
            "Я уехал в Корею, ключи отправлю почтой.",
        ],
        "offplatform": [
            "Звонки не принимаю, пишите только в Telegram.",
            "Связь только WhatsApp.",
            "Пишите в телеграм @{handle}.",
        ],
        "card": "Карта: {card}.",
        "generic": [
            "Отличная квартира, всё есть.",
            "Хорошая квартира для жизни, заходи и живи.",
            "Всё необходимое есть, отличная квартира.",
        ],
        "title": [
            "СРОЧНО {rooms}-комн., {district_ru}",
            "{rooms}-комн. квартира, {district_ru}",
            "Квартира {district_ru} недорого",
        ],
    },
    "uz": {
        "urgency": ["Shoshilinch!", "Faqat bugun shu narxda!", "Tezkor, xohlovchilar ko'p!"],
        "prepayment": [
            "Oldindan to'lov kartaga o'tkazing.",
            "Bron uchun avans 50$.",
            "Avval pul o'tkazing, keyin ko'rsataman.",
        ],
        "abroad": [
            "Hozir chet eldaman, kalitni pochta orqali yuboraman.",
            "Egasi xorijda, ko'rsatolmayman.",
        ],
        "offplatform": [
            "Faqat telegramga yozing.",
            "Qo'ng'iroq qilmang, faqat Telegram.",
            "Telegramga yozing: @{handle}.",
        ],
        "card": "Karta: {card}.",
        "generic": ["Zo'r kvartira, hammasi bor.", "Yaxshi kvartira, hammasi bor."],
        "title": ["Shoshilinch! {rooms} xonali, {district}", "{rooms} xonali kvartira, {district}"],
    },
    "en": {
        "urgency": ["URGENT!", "Today only at this price!", "Hurry, first come first served."],
        "prepayment": [
            "Deposit before viewing is required.",
            "Please send the money via Western Union to reserve.",
            "Advance payment of one month to hold the flat.",
        ],
        "abroad": [
            "I am currently abroad, keys will be sent by courier.",
            "I'm out of the country and can't show the flat.",
        ],
        "offplatform": [
            "No calls, message me on Telegram only.",
            "Contact only via WhatsApp.",
            "Telegram only: @{handle}.",
        ],
        "card": "Card: {card}.",
        "generic": ["Great apartment, everything you need.", "Nice flat, perfect place."],
        "title": ["URGENT {rooms}-bedroom in {district}", "Cheap apartment in {district}"],
    },
}

LANG_P = {"ru": 0.6, "uz": 0.25, "en": 0.15}
OPERATORS = ["90", "91", "93", "94", "97", "99", "33", "88", "95", "98"]


@dataclass
class SynthListing:
    listing: Listing
    hashes: list[PhotoHash]
    label: int
    lang: str


class _Gen:
    def __init__(self, seed: int, districts: DistrictTable) -> None:
        self.r = random.Random(seed)
        self.np = np.random.default_rng(seed)
        self.d = districts
        self.base_date = datetime(2026, 9, 1)

    # ---- примитивы
    def phone(self) -> str:
        r = self.r
        return (
            f"+998 {r.choice(OPERATORS)} {r.randint(100, 999)} {r.randint(10, 99)} "
            f"{r.randint(10, 99)}"
        )

    def card(self) -> str:
        r = self.r
        prefix = r.choice(["8600", "9860", "5614"])
        return " ".join([prefix] + [f"{r.randint(0, 9999):04d}" for _ in range(3)])

    def handle(self) -> str:
        return (
            self.r.choice(["arenda", "kvartira", "uy", "rent", "tashkent"])
            + "_"
            + str(self.r.randint(100, 9999))
        )

    def hash64(self) -> int:
        return int(self.np.integers(0, 2**64, dtype=np.uint64))

    def photo_set(self, k: int) -> list[PhotoHash]:
        return [PhotoHash(self.hash64(), self.hash64()) for _ in range(k)]

    def perturb(self, hs: list[PhotoHash], max_flips: int = 4) -> list[PhotoHash]:
        out = []
        for h in hs:
            ph, dh = h.phash, h.dhash
            for _ in range(self.r.randint(0, max_flips)):
                ph ^= 1 << self.r.randrange(64)
            for _ in range(self.r.randint(0, max_flips)):
                dh ^= 1 << self.r.randrange(64)
            out.append(PhotoHash(ph, dh))
        return out

    def lang(self) -> str:
        return self.r.choices(list(LANG_P), weights=list(LANG_P.values()))[0]

    def flat(self) -> dict:
        r = self.r
        district = r.choice(self.d.districts)
        rooms = r.choices([1, 2, 3, 4], weights=[0.3, 0.4, 0.2, 0.1])[0]
        area = max(18, round(rooms * r.uniform(17, 27) + r.uniform(8, 20)))
        floors = r.choice([4, 5, 9, 12, 16])
        return {
            "district": district,
            "district_ru": self.d.rows[district].name_ru,
            "rooms": rooms,
            "area": area,
            "floors": floors,
            "floor": r.randint(1, floors),
            "year": r.choice([2019, 2020, 2021, 2022, 2023, 2024, 2025]),
            "kitchen": r.randint(6, 16),
            "ceil": r.choice(["2,6", "2,7", "2,8", "3,0", "3,2"]),
            "minutes": r.randint(2, 20),
            "house": r.randint(1, 120),
            "street": r.choice(STREETS),
        }

    def fmt(self, s: str, ctx: dict, lang: str) -> str:
        lat = lang != "ru"
        return s.format(
            metro=self.r.choice(METRO_LAT if lat else METRO_RU),
            landmark=self.r.choice(LANDMARK_LAT if lat else LANDMARK_RU),
            handle=self.handle(),
            card=self.card(),
            **ctx,
        )

    def legit_text(self, ctx: dict, lang: str) -> list[str]:
        t = LEGIT[lang]
        r = self.r
        amen = {"ru": AMEN_RU, "uz": AMEN_UZ, "en": AMEN_EN}[lang]
        parts = [t["intro"]]
        parts += r.sample(t["location"], k=r.randint(1, 2))
        parts.append(r.choice(t["condition"]))
        parts += r.sample(t["details"], k=r.randint(1, 3))
        parts.append(t["amen"].replace("{amen}", ", ".join(r.sample(amen, k=r.randint(2, 5)))))
        parts += r.sample(t["terms"], k=r.randint(1, 3))
        parts.append(r.choice(t["viewing"]))
        return [self.fmt(p, ctx, lang) for p in parts]

    def signals(self, label: int, lang: str, ctx: dict) -> tuple[list[str], bool]:
        """Фразы-сигналы, выбранные по вероятностям P; второй элемент — шаблонный текст."""
        r = self.r
        idx = 0 if label else 1
        s = SCAM[lang]
        out = []
        for key in ("urgency", "prepayment", "abroad", "offplatform"):
            if r.random() < P[key][idx]:
                out.append(self.fmt(r.choice(s[key]), ctx, lang))
        if r.random() < P["card"][idx]:
            out.append(self.fmt(s["card"], ctx, lang))
        return out, r.random() < P["generic"][idx]

    def price(self, ctx: dict, label: int) -> tuple[float, str]:
        med = self.d.rows[ctx["district"]].median_usd_m2
        if label and self.r.random() >= P["market_price"][0]:
            mult = self.np.lognormal(np.log(0.62), 0.25)
        elif label:
            mult = self.np.lognormal(np.log(0.92), 0.15)
        else:
            mult = self.np.lognormal(0.0, 0.2)
        usd = max(60.0, round(med * mult * ctx["area"] / 10) * 10)
        if self.r.random() < 0.3:
            return round(usd * uzs_per_usd(), -4), "UZS"
        return usd, "USD"


def generate_listings(
    n: int = 2000, scam_ratio: float = 0.25, seed: int = 42, districts: DistrictTable | None = None
) -> list[SynthListing]:
    if not 0 < scam_ratio < 1:
        raise ValueError("scam_ratio должен быть в (0, 1)")
    districts = districts or DistrictTable.load()
    g = _Gen(seed, districts)
    r = g.r
    n_scam = round(n * scam_ratio)
    n_legit = n - n_scam
    scam_phones = [g.phone() for _ in range(max(5, n_scam // 8))]
    agency_phones = [g.phone() for _ in range(max(5, n_legit // 40))]
    stock_photos = [g.photo_set(r.randint(1, 8)) for _ in range(max(5, n_scam // 10))]

    legit: list[SynthListing] = []
    for i in range(n_legit):
        lang = g.lang()
        if legit and r.random() < P["repost"][1]:
            src = r.choice(legit)
            lst = src.listing.model_copy(update={"id": f"L{i:05d}", "posted_at": g.base_date})
            legit.append(SynthListing(lst, g.perturb(src.hashes, 2), 0, src.lang))
            continue
        ctx = g.flat()
        sig, generic = g.signals(0, lang, ctx)
        body = (
            [g.fmt(r.choice(SCAM[lang]["generic"]), ctx, lang)]
            if generic
            else g.legit_text(ctx, lang)
        )
        if not generic and r.random() < P["short_text"][1]:
            body = body[: r.randint(2, 4)]
        desc = " ".join(body + sig)
        price, cur = g.price(ctx, 0)
        phone = r.choice(agency_phones) if r.random() < P["shared_phone"][1] else g.phone()
        lst = Listing(
            id=f"L{i:05d}",
            title=g.fmt(r.choice(LEGIT[lang]["title"]), ctx, lang),
            description=desc,
            price=price,
            currency=cur,
            area_m2=ctx["area"],
            rooms=ctx["rooms"],
            district=ctx["district"] if lang != "ru" else ctx["district_ru"],
            contact_phone=phone,
            contact_name=r.choice(["Дилноза", "Азиз", "Sardor", "Malika", "Ольга", "Rustam"]),
            posted_at=g.base_date - timedelta(days=r.randint(0, 60)),
            source_url=f"https://example.local/rent/L{i:05d}",
        )
        legit.append(SynthListing(lst, g.photo_set(r.randint(1, 8)), 0, lang))

    scams: list[SynthListing] = []
    for j in range(n_scam):
        ctx = g.flat()
        victim = r.choice(legit) if legit else None
        lang = victim.lang if victim and r.random() < 0.5 else g.lang()
        sig, generic = g.signals(1, lang, ctx)
        if victim and r.random() < P["copy_text"][0]:
            lang = victim.lang
            sig, _ = g.signals(1, lang, ctx)
            desc = " ".join([victim.listing.description, *sig])
        else:
            body = (
                [g.fmt(r.choice(SCAM[lang]["generic"]), ctx, lang)]
                if generic
                else g.legit_text(ctx, lang)
            )
            if not generic and r.random() < P["short_text"][0]:
                body = body[: r.randint(2, 4)]
            desc = " ".join(body + sig)
        if r.random() < P["stolen_photos"][0]:
            if victim and r.random() < 0.5:
                k = r.randint(1, len(victim.hashes))
                hashes = g.perturb(victim.hashes[:k])
            else:
                hashes = g.perturb(r.choice(stock_photos))
        else:
            hashes = g.photo_set(r.randint(1, 8))
        price, cur = g.price(ctx, 1)
        phone = r.choice(scam_phones) if r.random() < P["shared_phone"][0] else g.phone()
        lst = Listing(
            id=f"S{j:05d}",
            title=g.fmt(r.choice(SCAM[lang]["title"]), ctx, lang),
            description=desc,
            price=price,
            currency=cur,
            area_m2=ctx["area"],
            rooms=ctx["rooms"],
            district=ctx["district"] if lang != "ru" else ctx["district_ru"],
            contact_phone=phone,
            contact_name=r.choice(["Алексей", "Aziz", "Owner", "Хозяин", "Jasur"]),
            posted_at=g.base_date - timedelta(days=r.randint(0, 10)),
            source_url=f"https://example.local/rent/S{j:05d}",
        )
        scams.append(SynthListing(lst, hashes, 1, lang))

    for item in legit + scams:  # шум разметки
        if r.random() < LABEL_NOISE:
            item.label = 1 - item.label

    out = legit + scams
    r.shuffle(out)
    return out


def make_dataset(
    n: int = 2000, scam_ratio: float = 0.25, seed: int = 42, districts: DistrictTable | None = None
) -> pd.DataFrame:
    """Датасет признаков: FEATURE_NAMES + listing_id, lang, label."""
    districts = districts or DistrictTable.load()
    items = generate_listings(n, scam_ratio, seed, districts)
    ctx = FeatureContext.from_listings(
        [x.listing for x in items], districts, photo_hashes={x.listing.id: x.hashes for x in items}
    )
    rows = []
    for x in items:
        values = ctx.features(x.listing).values
        rows.append({"listing_id": x.listing.id, "lang": x.lang, **values, "label": x.label})
    df = pd.DataFrame(rows)
    return df[["listing_id", "lang", *FEATURE_NAMES, "label"]]
