from __future__ import annotations

import pytest

from scam_detector import text_features as tf


@pytest.mark.parametrize(
    "text",
    [
        "СРОЧНО сдаю квартиру",
        "Только сегодня такая цена!",
        "Shoshilinch! arzon",
        "Faqat bugun shu narxda",
        "URGENT: flat for rent",
        "Hurry, first come first served",
    ],
)
def test_urgency_triggers(text):
    assert tf.urgency(text).count >= 1


@pytest.mark.parametrize(
    "text",
    [
        "Для брони нужна предоплата 50$",
        "Задаток до просмотра",
        "Переведите залог на карту",
        "Oldindan to'lov kartaga o'tkazing",
        "Bron uchun avans 50$",
        "Deposit before viewing is required",
        "Send the money via Western Union",
    ],
)
def test_prepayment_triggers(text):
    assert tf.prepayment(text).count >= 1


@pytest.mark.parametrize(
    "text",
    [
        "Я сейчас нахожусь за границей",
        "Хозяин живёт в Москве, показать не смогу",
        "ключи отправлю почтой",
        "Hozir chet eldaman",
        "Kalitni pochta orqali yuboraman",
        "I am currently abroad",
        "keys will be sent by courier",
    ],
)
def test_owner_abroad_triggers(text):
    assert tf.owner_abroad(text).count >= 1


@pytest.mark.parametrize(
    "text",
    [
        "Звонки не принимаю, пишите только в Telegram",
        "Пишите в WhatsApp",
        "Faqat telegramga yozing",
        "No calls, message me on Telegram",
        "Contact only via WhatsApp",
        "пишите @arenda_tash",
        "t.me/rent_tash",
    ],
)
def test_off_platform_triggers(text):
    assert tf.off_platform_contact(text).count >= 1


NEUTRAL = (
    "Сдаётся 2-комнатная квартира, Юнусабад, 62 м², 4/9 этаж. Рядом метро Минор. "
    "Евроремонт, есть кондиционер и стиральная машина. Залог за один месяц при подписании "
    "договора. Показ в любое время, звоните после 18:00."
)


def test_neutral_text_triggers_nothing():
    for fn in (tf.urgency, tf.prepayment, tf.owner_abroad, tf.off_platform_contact, tf.card_number):
        assert fn(NEUTRAL).count == 0, fn.__name__


def test_card_number():
    assert tf.card_number("Карта: 8600 1234 5678 9012").count == 1
    assert tf.card_number("Humo 9860-1111-2222-3333").count == 1
    assert tf.card_number("Телефон +998 90 111 22 33").count == 0


def test_generic_description():
    score, phrases = tf.generic_description("Отличная квартира, всё есть. Пишите.")
    assert score == 1.0 and phrases.count >= 1
    score_neutral, _ = tf.generic_description(NEUTRAL)
    assert score_neutral < 0.34


def test_uzbek_apostrophe_variants_are_normalized():
    assert tf.prepayment("Oldindan toʻlov").count == 1  # ʻ (U+02BB)
    assert tf.prepayment("Oldindan to’lov").count == 1  # ’ (U+2019)


def test_text_duplicate_jaccard():
    idx = tf.TextIndex().fit(["a", "b"], [NEUTRAL, "Совсем другой текст про дом в Сергели у парка"])
    copied = NEUTRAL + " СРОЧНО! Предоплата на карту."
    jac, other = idx.max_jaccard(copied, exclude_id="c")
    assert other == "a" and jac > 0.7
    jac_self, _ = idx.max_jaccard(NEUTRAL, exclude_id="a")
    assert jac_self < 0.2


def test_phone_reuse_across_listings_with_different_formatting():
    reg = tf.PhoneRegistry()
    reg.add("l1", "+998 90 111 22 33")
    reg.add("l2", "+998(90)111-22-33")
    reg.add("l3", "901112233")
    reg.add("l4", "+998 97 000 00 00")
    assert reg.reuse_count("+998 90 111 22 33", exclude_id="l1") == 2
    assert reg.reuse_count("+998 97 000 00 00", exclude_id="l4") == 0
    assert reg.reuse_count("") == 0
