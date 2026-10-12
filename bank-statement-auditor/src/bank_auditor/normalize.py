"""Нормализация: суммы, валюты, названия мерчантов, транслитерация ru<->lat."""

from __future__ import annotations

import re

# --------------------------------------------------------------------------- суммы

_MINUS_CHARS = "−‒–—"
_SPACE_CHARS = "    '`’"


def parse_amount(value: object) -> float | None:
    """Разобрать сумму из строки банковской выписки.

    Поддерживает: «125 000,00», «125,000.00», «1.234.567,89», «-1 234.5», «(99.90)»,
    неразрывные пробелы, юникодный минус, хвост/префикс валюты («UZS 15 000», «$9.99»).
    Правило неоднозначностей: если есть и «.», и «,» — десятичный разделитель тот,
    что стоит правее; одиночная «,» с ровно 3 цифрами после — разделитель тысяч,
    иначе — десятичная; одиночная «.» — всегда десятичная.
    Возвращает None для пустых/нечисловых значений.
    """
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "-", "—"}:
        return None
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative, s = True, s[1:-1]
    for ch in _MINUS_CHARS:
        s = s.replace(ch, "-")
    for ch in _SPACE_CHARS:
        s = s.replace(ch, "")
    # убрать валютные символы/буквы
    s = re.sub(r"[^\d.,+\-]", "", s)
    if s.startswith("-"):
        negative, s = not negative, s[1:]
    elif s.startswith("+"):
        s = s[1:]
    if s.endswith("-"):  # «150.00-» — так пишут некоторые банки
        negative, s = not negative, s[:-1]
    if not s or not re.search(r"\d", s):
        return None

    has_dot, has_comma = "." in s, "," in s
    if has_dot and has_comma:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif has_comma:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) != 3:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif has_dot and s.count(".") > 1:
        s = s.replace(".", "")
    try:
        number = float(s)
    except ValueError:
        return None
    return -number if negative else number


# --------------------------------------------------------------------------- валюты

_CURRENCY_ALIASES: dict[str, str] = {
    "UZS": "UZS",
    "SUM": "UZS",
    "SOM": "UZS",
    "SO'M": "UZS",
    "СУМ": "UZS",
    "СЎМ": "UZS",
    "СУММ": "UZS",
    "860": "UZS",
    "UZB": "UZS",
    "USD": "USD",
    "$": "USD",
    "US$": "USD",
    "840": "USD",
    "ДОЛЛ": "USD",
    "DOLLAR": "USD",
    "EUR": "EUR",
    "€": "EUR",
    "978": "EUR",
    "ЕВРО": "EUR",
    "EURO": "EUR",
    "RUB": "RUB",
    "RUR": "RUB",
    "₽": "RUB",
    "РУБ": "RUB",
    "643": "RUB",
    "KZT": "KZT",
    "₸": "KZT",
    "ТЕНГЕ": "KZT",
    "398": "KZT",
    "GBP": "GBP",
    "£": "GBP",
    "826": "GBP",
}


def normalize_currency(value: object, default: str = "UZS") -> str:
    """«сум», «so'm», «860», «$» → ISO-код. Неизвестное → 3 буквы как есть или default."""
    if value is None:
        return default
    s = str(value).strip().upper().replace("’", "'").replace("ʻ", "'")
    if not s or s == "NAN":
        return default
    s = s.rstrip(".")
    if s in _CURRENCY_ALIASES:
        return _CURRENCY_ALIASES[s]
    return s if re.fullmatch(r"[A-Z]{3}", s) else default


# --------------------------------------------------------------------------- транслитерация

_RU_TO_LAT: dict[str, str] = {
    "А": "A",
    "Б": "B",
    "В": "V",
    "Г": "G",
    "Д": "D",
    "Е": "E",
    "Ё": "YO",
    "Ж": "ZH",
    "З": "Z",
    "И": "I",
    "Й": "Y",
    "К": "K",
    "Л": "L",
    "М": "M",
    "Н": "N",
    "О": "O",
    "П": "P",
    "Р": "R",
    "С": "S",
    "Т": "T",
    "У": "U",
    "Ф": "F",
    "Х": "KH",
    "Ц": "TS",
    "Ч": "CH",
    "Ш": "SH",
    "Щ": "SCH",
    "Ъ": "",
    "Ы": "Y",
    "Ь": "",
    "Э": "E",
    "Ю": "YU",
    "Я": "YA",
    # узбекская кириллица
    "Ў": "O'",
    "Қ": "Q",
    "Ғ": "G'",
    "Ҳ": "H",
}

# обратная (латиница -> кириллица): сначала многобуквенные сочетания
_LAT_TO_RU_MULTI: list[tuple[str, str]] = [
    ("SHCH", "Щ"),
    ("SCH", "Щ"),
    ("YO", "Ё"),
    ("ZH", "Ж"),
    ("KH", "Х"),
    ("TS", "Ц"),
    ("CH", "Ч"),
    ("SH", "Ш"),
    ("YU", "Ю"),
    ("YA", "Я"),
    ("O'", "Ў"),
    ("G'", "Ғ"),
]
_LAT_TO_RU_SINGLE: dict[str, str] = {
    "A": "А",
    "B": "Б",
    "C": "К",
    "D": "Д",
    "E": "Е",
    "F": "Ф",
    "G": "Г",
    "H": "Х",
    "I": "И",
    "J": "ДЖ",
    "K": "К",
    "L": "Л",
    "M": "М",
    "N": "Н",
    "O": "О",
    "P": "П",
    "Q": "К",
    "R": "Р",
    "S": "С",
    "T": "Т",
    "U": "У",
    "V": "В",
    "W": "В",
    "X": "КС",
    "Y": "Й",
    "Z": "З",
}


def translit_ru_to_lat(text: str) -> str:
    """Базовая транслитерация кириллицы в латиницу (результат в верхнем регистре)."""
    return "".join(_RU_TO_LAT.get(ch, ch) for ch in text.upper())


def translit_lat_to_ru(text: str) -> str:
    """Базовая обратная транслитерация латиницы в кириллицу (верхний регистр)."""
    s = text.upper()
    out: list[str] = []
    i = 0
    while i < len(s):
        for lat, ru in _LAT_TO_RU_MULTI:
            if s.startswith(lat, i):
                out.append(ru)
                i += len(lat)
                break
        else:
            out.append(_LAT_TO_RU_SINGLE.get(s[i], s[i]))
            i += 1
    return "".join(out)


def has_cyrillic(text: str) -> bool:
    return bool(re.search(r"[Ѐ-ӿ]", text))


# --------------------------------------------------------------------------- мерчанты

# после транслитерации кириллические написания брендов приводим к «каноническим»
_BRAND_ALIASES: dict[str, str] = {
    "YANDEKS": "YANDEX",
    "PLYUS": "PLUS",
    "NETFLIKS": "NETFLIX",
    "SPOTIFAY": "SPOTIFY",
    "EPPL": "APPLE",
    "AYKLAUD": "ICLOUD",
    "BILAYN": "BEELINE",
    "BILAIN": "BEELINE",
    "MEGOGO": "MEGOGO",
    "TAKSI": "TAXI",
}

_OPERATION_PREFIXES = (
    r"POKUPKA|OPLATA|SPISANIE|PURCHASE|PAYMENT|POS|TO'LOV|TOLOV|XARID|"
    r"OPERATSIYA|PLATEZH|BEZNALICHNAYA"
)
_LEGAL_FORMS = r"OOO|LLC|MCHJ|XK|QK|AO|JSC|PAO|ZAO|IP|CHP|LTD|INC|GMBH|BV"
_LOCATIONS = (
    r"TASHKENT|TOSHKENT|SAMARKAND|SAMARQAND|BUKHARA|BUXORO|ANDIJAN|NAMANGAN|"
    r"FERGANA|NUKUS|MOSCOW|MOSKVA|ALMATY|AMSTERDAM|LONDON|DUBLIN|"
    r"LUXEMBOURG|STOCKHOLM|CUPERTINO|LOS GATOS|LIMASSOL|NICOSIA"
)
_COUNTRY_CODES = r"UZ|UZB|NL|GB|GBR|IE|IRL|US|USA|RU|RUS|KZ|LU|SE|CY|DE"

_CARD_MASK = re.compile(r"\b\d{4}[\s-]?\d{0,2}[*X•]{2,}[\s*X•\d-]*?\d{4}\b")
_CARD_WORD = re.compile(r"(КАРТА|KARTA|CARD)\s*[*#№]?\s*\d{4}")
_CARD_TAIL = re.compile(r"[*•]+\d{4}\b")
_DATE_RE = re.compile(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}-\d{2}-\d{2})\b")
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(:\d{2})?\b")
_TXN_ID = re.compile(r"\b(ID|TXN|REF|RRN|AUTH|ЧЕК|CHEK|NO)\s*[:#№]?\s*[A-Z0-9-]*\d[A-Z0-9-]{3,}")
_NUM_SIGN = re.compile(r"№\s*[A-Z0-9-]+")
_LONG_DIGITS = re.compile(r"\b[A-Z]*\d{5,}[A-Z0-9]*\b")


def normalize_merchant(description: str) -> str:
    """Привести описание операции к «чистому» имени мерчанта.

    Убирает маски карт, ID транзакций, даты/время, правовые формы (ООО/LLC/MCHJ),
    города и коды стран, доменные хвосты; переводит кириллицу в латиницу и
    сводит распространённые бренды к одному написанию. Результат — UPPER CASE.

    >>> normalize_merchant("NETFLIX.COM 4169****1234 AMSTERDAM NL")
    'NETFLIX'
    """
    s = description.upper().replace("Ё", "Е")
    s = _CARD_MASK.sub(" ", s)
    s = _CARD_WORD.sub(" ", s)
    s = _CARD_TAIL.sub(" ", s)
    s = _DATE_RE.sub(" ", s)
    s = _TIME_RE.sub(" ", s)
    s = _TXN_ID.sub(" ", s)
    s = _NUM_SIGN.sub(" ", s)
    s = _LONG_DIGITS.sub(" ", s)
    s = translit_ru_to_lat(s)
    s = re.sub(r"\bWWW\.", " ", s)
    s = re.sub(r"\.(COM|RU|UZ|NET|ORG|IO|TV)\b", " ", s)
    # сохранить узбекские o'/g' при чистке пунктуации
    s = s.replace("O'", "O\u0001").replace("G'", "G\u0001")
    s = re.sub(r"[^A-Z0-9\u0001&+ ]", " ", s)
    s = s.replace("\u0001", "'")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(rf"^(?:(?:{_OPERATION_PREFIXES})\b\s*)+", "", s)
    s = re.sub(rf"\b(?:{_LEGAL_FORMS})\b", " ", s)
    s = re.sub(rf"\b(?:{_LOCATIONS})\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(rf"(?:\s+\b(?:{_COUNTRY_CODES})\b)+$", "", s)
    tokens = [_BRAND_ALIASES.get(tok, tok) for tok in s.split()]
    while len(tokens) > 1 and re.fullmatch(r"\d{1,4}", tokens[-1]):
        tokens.pop()
    result = " ".join(tokens).strip()
    return result or description.strip().upper()


def merchant_root(merchant: str) -> str:
    """Первый значимый токен имени — для поиска «связанной активности»."""
    for tok in merchant.split():
        if len(tok) >= 4 and not tok.isdigit():
            return tok
    return merchant
