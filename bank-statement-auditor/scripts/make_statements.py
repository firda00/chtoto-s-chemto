"""Генератор синтетических банковских выписок (детерминированный, seed=42).

Одни и те же ~300 операций за 6 месяцев (апрель–сентябрь 2026) записываются в три
разных CSV-формата + маленький PDF:

* statement_uzbank.csv  — «узбекский» банк: cp1251, `;`, date;description;debit;credit;balance,
                          даты 05.04.2026, суммы «126 300,00», шапка и строка «Итого»;
* statement_rubank.csv  — «российский» формат: utf-8, `,`, Дата,Операция,Сумма,Валюта,
                          даты «2026-04-05 14:32», знаковая сумма, валюта «сум»;
* statement_sample.csv  — generic English: utf-8, `,`, Posted Date,Description,Amount,
                          Currency,Reference, даты MM/DD/YYYY, «-126,300.00», новые сверху;
* statement_small.pdf   — апрельские операции таблицей (pdfplumber), нужен reportlab.

Заложенные паттерны: Netflix, Spotify (+ двойное списание в июле), iCloud, Яндекс Плюс
(повышение цены 29 900 → 39 900), Beeline, Uztelecom, забытый фитнес-клуб (ходил только
в апреле–мае), MEGOGO (пробное списание 1 000 → платная), IVI (отменена в июне),
всплеск — ноутбук в Texnomart.

Запуск:  uv run python scripts/make_statements.py [--out data]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import random
from dataclasses import dataclass
from pathlib import Path

START = dt.date(2026, 4, 1)
END = dt.date(2026, 9, 30)
MONTHS = [(2026, m) for m in range(4, 10)]
OPENING_BALANCE = 4_000_000.0


@dataclass
class Event:
    date: dt.date
    time: str
    description: str
    amount: float  # знаковая: < 0 списание, > 0 зачисление

    @property
    def key(self) -> tuple:
        direction = "debit" if self.amount < 0 else "credit"
        return (self.date, direction, abs(self.amount), self.description)


def _day(year: int, month: int, day: int) -> dt.date:
    return dt.date(year, month, min(day, 28 if month == 2 else 30))


def build_events(seed: int = 42) -> list[Event]:
    rng = random.Random(seed)
    events: list[Event] = []

    def add(date: dt.date, desc: str, amount: float) -> None:
        t = f"{rng.randint(7, 22):02d}:{rng.randint(0, 59):02d}"
        events.append(Event(date, t, desc, round(amount, 2)))

    def noisy(base: float, pct: float = 0.015) -> float:
        return round(base * (1 + rng.uniform(-pct, pct)), -2)

    def rand_amount(lo: int, hi: int, step: int = 100) -> float:
        return float(rng.randrange(lo, hi, step))

    def rrn() -> str:
        return str(rng.randint(10_000_000, 99_999_999))

    # ---------------------------------------------------------------- регулярные
    for y, m in MONTHS:
        add(_day(y, m, 25), "ЗАРПЛАТА ООО DIGITAL SOLUTIONS", 12_500_000)
        add(
            _day(y, m, 5 + rng.choice([-1, 0, 1])),
            f"NETFLIX.COM 4169****7788 RRN {rrn()} AMSTERDAM NL",
            -noisy(126_000),
        )
        spotify_day = _day(y, m, 8 + rng.choice([0, 1]))
        spotify_amount = noisy(75_000)
        add(spotify_day, f"SPOTIFY ID {rrn()} STOCKHOLM SE", -spotify_amount)
        if m == 7:  # двойное списание
            events.append(
                Event(spotify_day, events[-1].time, events[-1].description, -spotify_amount)
            )
        add(_day(y, m, 14), "APPLE.COM/BILL ICLOUD CUPERTINO US", -noisy(12_500, 0.01))
        add(
            _day(y, m, 20 + rng.choice([0, 1])),
            "ЯНДЕКС.ПЛЮС МОСКВА",
            -(29_900 if m <= 6 else 39_900),
        )
        add(_day(y, m, 1 + rng.choice([0, 1])), "PAYNET BEELINE 998901234567", -85_000)
        add(_day(y, m, 18), "UZTELECOM INTERNET SHARTNOMA 4455123", -150_000)
        add(_day(y, m, 3), "ATLETIK FITNESS CLUB TOSHKENT", -350_000)
        add(
            _day(y, m, 12),
            "HUDUDGAZ TOSHKENT TO'LOV",
            -rand_amount(*{4: (140_000, 160_000), 5: (90_000, 110_000)}.get(m, (50_000, 75_000))),
        )
        add(_day(y, m, 10), "HUDUDIY ELEKTR TARMOQLARI", -rand_amount(90_000, 150_000))

    # MEGOGO: пробное списание, затем платная подписка
    add(dt.date(2026, 5, 12), "MEGOGO.NET LIMASSOL CY", -1_000)
    for m in (6, 7, 8, 9):
        add(dt.date(2026, m, 12), "MEGOGO.NET LIMASSOL CY", -49_000)
    # IVI: оформлена и отменена в июне
    for m in (4, 5, 6):
        add(dt.date(2026, m, 15), "IVI.RU MOSCOW RU", -39_000)

    # спортзал: активность только в апреле–мае, потом — только абонемент
    for d in sorted(rng.sample(range(4, 50), 7)):
        date = START + dt.timedelta(days=d)
        add(date, "ATLETIK FITNESS CLUB TOSHKENT", -rand_amount(20_000, 46_000, 1000))
    for d in (dt.date(2026, 4, 9), dt.date(2026, 4, 23)):
        add(d, "ATLETIK FITNESS PERSONAL TRAINING", -150_000)

    # ---------------------------------------------------------------- повседневное
    days = (END - START).days + 1

    def scatter(n: int, descs: list[str], lo: int, hi: int, step: int = 100) -> None:
        for _ in range(n):
            date = START + dt.timedelta(days=rng.randrange(days))
            add(date, rng.choice(descs), -rand_amount(lo, hi, step))

    scatter(
        75, ["KORZINKA CHILONZOR TOSHKENT", "MAKRO YUNUSOBOD", "HAVAS FOOD 12"], 40_000, 450_000
    )
    scatter(
        45,
        [
            "EVOS 4169****7788",
            "OQTEPA LAVASH",
            "SAFIA BAKERY",
            "КАФЕ РОХАТ",
            "COFFEE HOUSE MAGIC CITY",
        ],
        25_000,
        180_000,
    )
    scatter(55, ["YANDEX GO TAXI", "YANDEX GO TAXI", "MYTAXI"], 12_000, 65_000)
    scatter(
        14,
        ["ПЕРЕВОД НА КАРТУ 8600****1122", "P2P O'TKAZMA HUMO 9860****3344"],
        200_000,
        1_500_000,
        10_000,
    )
    scatter(
        8,
        ["ATM KAPITALBANK CHILONZOR", "СНЯТИЕ НАЛИЧНЫХ БАНКОМАТ 0451"],
        300_000,
        1_000_000,
        50_000,
    )
    scatter(10, ["APTEKA 999 TOSHKENT", "DORIXONA GRAND PHARM"], 15_000, 220_000)
    scatter(8, ["MAGIC CITY PARK", "CINEMAX KINOTEATR", "BOWLING CITY"], 50_000, 300_000)
    scatter(15, ["UZUM MARKET", "ZARA SAMARQAND DARVOZA", "MEDIAPARK"], 90_000, 900_000)
    scatter(5, ["IP KARIMOV A.A.", "OOO ZIYO NUR"], 30_000, 250_000)
    for _ in range(4):
        date = START + dt.timedelta(days=rng.randrange(days))
        add(date, "P2P PEREVOD OT 8600****5566", rand_amount(100_000, 800_000, 10_000))

    # всплеск: ноутбук
    add(dt.date(2026, 8, 16), "TEXNOMART SERGELI", -8_900_000)

    # гарантируем: никаких случайных «дублей», кроме заложенного Spotify
    seen: set[tuple] = set()
    for e in events:
        k = (e.date, e.description, e.amount)
        while k in seen and not e.description.startswith("SPOTIFY"):
            e.amount -= 100 if e.amount < 0 else -100
            k = (e.date, e.description, e.amount)
        seen.add(k)

    events.sort(key=lambda e: e.key)
    return events


# ---------------------------------------------------------------- форматтеры сумм


def fmt_space_comma(x: float) -> str:
    """126300 -> '126 300,00'"""
    return f"{x:,.2f}".replace(",", " ").replace(".", ",")


def fmt_plain(x: float) -> str:
    return f"{x:.2f}"


def fmt_us(x: float) -> str:
    return f"{x:,.2f}"


# ---------------------------------------------------------------- writers


def write_uzbank(events: list[Event], path: Path) -> None:
    buf = io.StringIO()
    buf.write("АКБ «Namuna Bank» — выписка по карточному счёту 20206000900000001;;;;\n")
    buf.write("Период: 01.04.2026 - 30.09.2026;;;;\n")
    w = csv.writer(buf, delimiter=";", lineterminator="\n")
    w.writerow(["date", "description", "debit", "credit", "balance"])
    balance = OPENING_BALANCE
    total_debit = total_credit = 0.0
    for e in events:
        balance += e.amount
        debit = fmt_space_comma(-e.amount) if e.amount < 0 else ""
        credit = fmt_space_comma(e.amount) if e.amount > 0 else ""
        total_debit += max(0.0, -e.amount)
        total_credit += max(0.0, e.amount)
        w.writerow(
            [e.date.strftime("%d.%m.%Y"), e.description, debit, credit, fmt_space_comma(balance)]
        )
    w.writerow(["Итого", "", fmt_space_comma(total_debit), fmt_space_comma(total_credit), ""])
    path.write_bytes(buf.getvalue().encode("cp1251"))


def write_rubank(events: list[Event], path: Path) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=",", lineterminator="\n")
    w.writerow(["Дата", "Операция", "Сумма", "Валюта"])
    for e in events:
        w.writerow([f"{e.date.isoformat()} {e.time}", e.description, fmt_plain(e.amount), "сум"])
    path.write_text(buf.getvalue(), encoding="utf-8")


def write_generic(events: list[Event], path: Path) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=",", lineterminator="\n")
    w.writerow(["Posted Date", "Description", "Amount", "Currency", "Reference"])
    for i, e in reversed(list(enumerate(events, start=1))):  # новые сверху
        w.writerow(
            [e.date.strftime("%m/%d/%Y"), e.description, fmt_us(e.amount), "UZS", f"TX{i:06d}"]
        )
    path.write_text(buf.getvalue(), encoding="utf-8")


def pdf_events(events: list[Event]) -> list[Event]:
    """Апрельские операции с латинскими описаниями (у стандартных PDF-шрифтов нет кириллицы)."""
    return [e for e in events if e.date.month == 4 and e.description.isascii()]


def write_pdf(events: list[Event], path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    rows = [["Date", "Description", "Amount", "Currency"]]
    for e in pdf_events(events):
        amount = f"{e.amount:,.2f}".replace(",", " ")
        rows.append([e.date.strftime("%d.%m.%Y"), e.description, amount, "UZS"])
    table = Table(rows, repeatRows=1, colWidths=[70, 280, 90, 50])
    table.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Helvetica", 8),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("ALIGN", (2, 1), (2, -1), "RIGHT"),
            ]
        )
    )
    doc = SimpleDocTemplate(str(path), pagesize=A4, title="Namuna Bank statement", invariant=1)
    doc.build(
        [
            Paragraph("Namuna Bank - card account statement, April 2026", styles["Title"]),
            Spacer(1, 8),
            table,
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pdf", action="store_true", help="не генерировать PDF")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    events = build_events(args.seed)
    write_uzbank(events, args.out / "statement_uzbank.csv")
    write_rubank(events, args.out / "statement_rubank.csv")
    write_generic(events, args.out / "statement_sample.csv")
    if not args.no_pdf:
        write_pdf(events, args.out / "statement_small.pdf")
    print(f"{len(events)} transactions -> {args.out}")


if __name__ == "__main__":
    main()
