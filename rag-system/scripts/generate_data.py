"""Генерирует синтетическую базу знаний вымышленного коворкинга Orbita Hub.

Запуск:  uv run python scripts/generate_data.py
Пишет:   data/docs/*.md (7 документов, RU + EN) и data/eval.jsonl (вопрос -> ожидаемый ответ).
Все данные вымышленные.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "docs"
EVAL_PATH = ROOT / "data" / "eval.jsonl"

DOCS: dict[str, str] = {
    "pricing.md": """---
title: Тарифы Orbita Hub
collection: pricing
lang: ru
---
# Тарифы Orbita Hub

Все цены указаны в сумах (UZS) и включают НДС. Оплата картой Uzcard, Humo, Visa или переводом по счёту.

## Гостевой день (Day Pass)

Гостевой день стоит 120 000 сум. В стоимость входят место в открытой зоне с 9:00 до 21:00, Wi-Fi, чай и фильтр-кофе.

## Нефиксированное место (Flex)

Абонемент Flex стоит 1 400 000 сум в месяц. Можно садиться за любой свободный стол в открытой зоне.
В тариф включены 4 часа переговорной в месяц.

## Фиксированное место (Fixed Desk)

Фиксированный стол стоит 1 900 000 сум в месяц. За резидентом закреплён стол, тумба с замком и доступ 24/7.

## Приватный офис

Офис на 4 человека стоит 7 500 000 сум в месяц, офис на 8 человек стоит 13 800 000 сум в месяц.
Минимальный срок аренды офиса составляет 3 месяца.

## Переговорные комнаты

Переговорная «Луна» (до 6 человек) стоит 150 000 сум в час.
Конференц-зал «Сатурн» (до 30 человек) стоит 450 000 сум в час.

## Скидки

Студентам очной формы обучения положена скидка 20% на Flex и Day Pass при предъявлении студенческого билета.
При оплате за 6 месяцев вперёд действует скидка 10% на любой тариф. Скидки не суммируются.
""",
    "schedule.md": """---
title: Режим работы и расписание
collection: schedule
lang: ru
---
# Режим работы и расписание

## Часы работы

Открытая зона работает с понедельника по пятницу с 9:00 до 21:00, в субботу с 10:00 до 18:00.
Воскресенье — выходной день.
Резиденты тарифов Fixed Desk и «Приватный офис» имеют доступ 24/7 по электронной карте.

## Ресепшен

Ресепшен работает по будням с 9:00 до 19:00. В субботу гостей встречает дежурный администратор.

## Праздничные дни

В государственные праздники (Навруз 21 марта, День независимости 1 сентября) открытая зона закрыта.
Доступ 24/7 для резидентов в праздники сохраняется.

## Регулярные мероприятия

Нетворкинг для резидентов проходит каждый четверг в 19:00 в лаунж-зоне.
Открытые лекции Orbita Talks проходят в первую субботу месяца в 12:00.
Участие в лекциях бесплатное для резидентов, для гостей стоит 50 000 сум.
""",
    "faq.md": """---
title: Frequently Asked Questions
collection: faq
lang: en
---
# Frequently Asked Questions

## What is the Wi-Fi speed?

The coworking has a 500 Mbps fiber connection with a backup 4G line. The guest network is called Orbita-Guest.

## Can I bring a guest?

Residents can bring up to 2 guests per day for free for up to 2 hours. Longer visits require a Day Pass.

## Is there printing?

Black-and-white printing costs 1 000 UZS per page. Color printing costs 3 000 UZS per page. Scanning is free.

## Is parking available?

There are 12 free parking spaces in the courtyard on a first-come basis. Bicycle parking is at the entrance.

## Do you have lockers?

Personal lockers can be rented for 100 000 UZS per month. Fixed Desk residents get a lockable drawer included.

## Is there a kitchen?

A shared kitchen with a fridge, a microwave and a coffee machine is on the 2nd floor.
""",
    "policies.md": """---
title: House Rules and Policies
collection: policies
lang: en
---
# House Rules and Policies

## Refund policy

A monthly membership can be refunded in full within 7 days of purchase if it has not been used.
After 7 days refunds are not available, but a membership can be frozen once for up to 14 days.

## Meeting room cancellation

Meeting room bookings can be cancelled free of charge up to 24 hours before the start.
Late cancellations are charged 50% of the booking price.

## Pets

Dogs under 10 kg are welcome on Fridays in the open area. Pets are not allowed in meeting rooms or the kitchen.

## Quiet zone

The quiet zone on the 3rd floor is for focused work: no phone calls and no video meetings.
Calls should be taken in the phone booths.

## Security

CCTV operates in common areas. A lost access card is replaced for 50 000 UZS.
""",
    "about.md": """---
title: О компании Orbita Hub
collection: company
lang: ru
---
# О компании Orbita Hub

Orbita Hub — вымышленная сеть коворкингов. Компания основана в 2020 году в Ташкенте.

## Площадка

Главная площадка находится по адресу: Ташкент, ул. Кометная, 7. Площадь коворкинга 1 200 м², три этажа.
На первом этаже открытая зона и лаунж, на втором кухня и переговорные, на третьем тихая зона и офисы.

## Команда

В команде 14 человек: администраторы, комьюнити-менеджеры и служба эксплуатации.

## Планы

Открытие второй площадки в районе Ташкент-Сити запланировано на март 2027 года.
""",
    "services.md": """---
title: Additional Services
collection: services
lang: en
---
# Additional Services

## Legal address

Registration of a legal address for a company costs 600 000 UZS per month, with a minimum term of 6 months.

## Mail handling

We receive letters and parcels for residents and notify them in Telegram.
Mail handling is free for Fixed Desk and Office residents and costs 80 000 UZS per month for Flex members.

## Event hosting

The Saturn hall can be booked for evening events after 19:00. Catering is provided by the partner cafe Kometa.

## Coffee bar

Specialty coffee at the bar costs 25 000 UZS per cup. Filter coffee and tea in the kitchen are free.
""",
    "support.md": """---
title: Поддержка и контакты
collection: company
lang: ru
---
# Поддержка и контакты

## Как связаться

Почта поддержки: support@orbita-hub.example. Телеграм-бот: @orbita_hub_bot.

## Сроки ответа

Поддержка отвечает на обращения в течение 2 часов в рабочее время и в течение 12 часов в выходные.

## Технические проблемы

Если не работает Wi-Fi или электронная карта, обратитесь к администратору на ресепшене
или напишите в бот с пометкой «срочно».
""",
}

EVAL: list[dict[str, str]] = [
    {"question": "Сколько стоит гостевой день?", "expected": "120 000", "source": "pricing.md"},
    {"question": "How much does color printing cost?", "expected": "3 000", "source": "faq.md"},
    {
        "question": "До скольки работает коворкинг в субботу?",
        "expected": "18:00",
        "source": "schedule.md",
    },
    {
        "question": "Can I get a refund for a monthly membership?",
        "expected": "7 days",
        "source": "policies.md",
    },
    {"question": "Какая скидка положена студентам?", "expected": "20%", "source": "pricing.md"},
    {
        "question": "Are dogs allowed in the coworking?",
        "expected": "10 kg",
        "source": "policies.md",
    },
    {
        "question": "Когда проходит нетворкинг для резидентов?",
        "expected": "четверг",
        "source": "schedule.md",
    },
    {
        "question": "How much is legal address registration?",
        "expected": "600 000",
        "source": "services.md",
    },
    {"question": "В каком году основана компания?", "expected": "2020", "source": "about.md"},
    {"question": "Как быстро отвечает поддержка?", "expected": "2 часов", "source": "support.md"},
    {"question": "What is the Wi-Fi speed?", "expected": "500 Mbps", "source": "faq.md"},
    {
        "question": "Сколько стоит переговорная Луна в час?",
        "expected": "150 000",
        "source": "pricing.md",
    },
]


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for name, text in DOCS.items():
        (DOCS_DIR / name).write_text(text, encoding="utf-8")
    with EVAL_PATH.open("w", encoding="utf-8") as fh:
        for row in EVAL:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(DOCS)} docs to {DOCS_DIR} and {len(EVAL)} eval rows to {EVAL_PATH}")


if __name__ == "__main__":
    main()
