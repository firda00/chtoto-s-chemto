# Local Rental Listing Scam Detector

Пайплайн проверки объявлений аренды жилья (Ташкент): поиск дубликатов фото по перцептивным
хэшам, анализ текста на признаки мошенничества (RU / UZ / EN) и расчёт риска — **Scam Score 0–100**
через CatBoost с объяснением каждого балла (SHAP-вклады + причины на русском).

LLM и сеть не используются: признаки — правила, регулярки и статистика, модель — CatBoost.

## Пайплайн

```
 listing.json ──► Listing (Pydantic)
                     │
     ┌───────────────┼──────────────────────────┬─────────────────────────────┐
     ▼               ▼                          ▼                             ▼
 images.py      text_features.py          text_features.py               корпус объявлений
 pHash+dHash    регэкспы RU/UZ/EN:        цена $/м² vs медиана           (папка с JSON / датасет)
 каждого фото   срочность, предоплата,    района (district_prices.csv)   ├─ PhotoIndex: те же фото
     │          «хозяин за границей»,     → ratio, z-score               │  в других объявлениях
     │          Telegram/WhatsApp only,   UZS→USD                        ├─ TextIndex: Jaccard по
     │          номер карты, шаблонность                                 │  шинглам (копия текста)
     │               │                          │                        └─ PhoneRegistry: телефон
     └───────────────┴──────────────┬───────────┴─────────────────────────────┘  в N объявлениях
                                    ▼
                      features.py → 22 признака (21 число + район)
                                    ▼
                  CatBoostClassifier (models/scam.cbm) → P(scam)
                                    ▼
          predict.py: Scam Score = P×100, уровень риска, SHAP (ShapValues)
                      → человекочитаемые причины → ScamReport
                                    ▼
                  CLI (typer + rich)  /  FastAPI POST /check
```

## Признаки

| Группа | Признаки | Как считается |
|---|---|---|
| Цена | `price_usd_m2`, `price_ratio_to_median`, `price_z`, `too_good_price`, `district_known` | цена/мес → USD (курс `SCAM_UZS_PER_USD`) / площадь; сравнение с медианой и std района из `data/district_prices.csv` (12 районов, русские/узбекские/латинские алиасы). `too_good` = ниже 65% медианы. Неизвестный район → средние по городу |
| Фото | `n_photos`, `photo_dup_count`, `photo_dup_listings`, `photo_dup_diff_district`, `photo_dup_price_diff` | pHash и dHash (64 бита, `imagehash`); дубликат, если `max(hamming_pHash, hamming_dHash) ≤ 10`. Поиск в `PhotoIndex` векторизован (numpy xor + popcount). Паттерн «те же фото, другой район/цена» — отдельные признаки |
| Текст | `urgency_hits`, `prepayment_hits`, `owner_abroad_hits`, `offplatform_hits`, `card_number` | регэкспы RU / UZ (латиница и кириллица, все варианты апострофа `ʻ’'`) / EN: «срочно», «faqat bugun», «today only»; «предоплата», «задаток до просмотра», «oldindan to'lov», «Western Union»; «нахожусь за границей», «kalitni pochta orqali», «keys by courier»; «только Telegram», «звонки не принимаю», `t.me/…`, `@handle`; номера карт Uzcard/Humo/Visa/MC |
| Шаблонность | `generic_score`, `desc_words` | доля из трёх флагов: < 25 слов; нет конкретики (этаж, метро, ориентир, цифры); общие фразы «отличная квартира, всё есть», «hammasi bor», «great apartment» |
| Корпус | `text_dup_jaccard`, `phone_reuse` | max Jaccard по словесным 3-граммам с другими объявлениями (ниже 0.5 обнуляется — это общая лексика, а не копия); число других объявлений с тем же телефоном (нормализация до 9 цифр) |
| Категория | `district` | канонический район, в CatBoost — one-hot |

## Быстрый старт

```bash
cd rental-scam-detector
uv sync

uv run scam make-dataset                          # data/dataset.csv: 2000 строк, 25% scam
uv run scam train                                 # models/scam.cbm + metrics.json + feature_importance.json
uv run scam check data/samples/listing_02.json    # отчёт по одному объявлению (--json для JSON)
uv run scam batch data/samples/                   # таблица по папке (--out reports.json)
uv run scam serve                                 # FastAPI на http://127.0.0.1:8000, POST /check

uv run pytest -q
uv run ruff check .
```

Параметры генератора: `uv run scam make-dataset --n 5000 --scam-ratio 0.1 --seed 1`
(то же самое — `uv run python scripts/make_dataset.py ...`). Демо-объявления и картинки
пересоздаются `uv run python scripts/make_samples.py`.

API:

```bash
curl -X POST http://127.0.0.1:8000/check -H "Content-Type: application/json" \
     -d @data/samples/listing_04.json
```

Корпус для поиска дубликатов в API — `SCAM_CORPUS_DIR` (по умолчанию `data/samples`). Пути к фото
в теле запроса должны быть доступны серверу; без фото проверка тоже работает.

## Метрики на синтетике

`uv run scam make-dataset && uv run scam train` (seed 42, 2000 объявлений, 26.1% scam с учётом
шума разметки, стратифицированный holdout 20% = 400 объявлений, CatBoost depth 4, 400 итераций):

| ROC-AUC | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|
| **0.943** | 0.787 | 0.946 | 0.673 | 0.905 |

Порог 0.5; recall можно поднять, снизив порог (уровни риска в отчёте: < 25 low, < 50 medium,
< 75 high, иначе critical). Обучение — около 1 секунды.

Топ важностей (PredictionValuesChange): `urgency_hits` 12.0, `price_z` 10.5,
`price_ratio_to_median` 9.9, `phone_reuse` 8.1, `owner_abroad_hits` 6.8, `area_m2` 5.6,
`text_dup_jaccard` 5.6, `prepayment_hits` 5.1, `offplatform_hits` 5.1, `desc_words` 4.4.
Заметная важность `area_m2` и `district` — подгонка под шум: в генераторе они с меткой не связаны.

Демо-объявления (`uv run scam batch data/samples/`):

| Объявление | Что это | Score | Риск |
|---|---|---|---|
| listing_02 | RU-мошенник: перезалитое фото listing_01, Мирабад за полцены, предоплата на карту, «за границей», только Telegram | 99.9 | critical |
| listing_04 | UZ-мошенник: «Shoshilinch», «chet eldaman», «oldindan to'lov», тот же телефон, что у listing_02 | 99.0 | critical |
| listing_06 | пограничное: шаблонный текст, только WhatsApp, цена на 31% ниже медианы | 35.7 | medium |
| listing_01 | честное объявление, у которого украли фото | 3.3 | low |
| listing_05 | честное EN-объявление | 2.4 | low |
| listing_03 | честное объявление, цена в UZS | 1.0 | low |

## Пример отчёта

`uv run scam check data/samples/listing_02.json`:

```
listing_02  Scam Score: 99.9/100 (critical)
  - Легенда «хозяин далеко, покажет курьер»: «за границей», «ключи передам через курьера»
  - Цена 4.2 $/м² на 56% ниже медианы района Мирабад (9.5 $/м²)
  - Давление срочностью: «срочно»
  - В тексте указан номер банковской карты
  - Контакт только вне площадки: «звонки не принимаю», «только в telegram», «@arenda_tash»
  - Просьба о предоплате/переводе до просмотра: «предоплата»
  - 1 фото совпадают с объявлением listing_01 (другой район: Юнусабад; цена отличается на 46%), расстояние хэшей 4 бит
  - Тот же телефон ещё в других объявлениях: 1
   Топ-вклады признаков (SHAP, log-odds)
 Признак               │ Значение │  Вклад
───────────────────────┼──────────┼────────
 owner_abroad_hits     │     2.00 │ +1.608
 price_ratio_to_median │     0.44 │ +1.260
 urgency_hits          │     2.00 │ +1.083
 price_z               │    -2.67 │ +0.809
 card_number           │     1.00 │ +0.807
 offplatform_hits      │     3.00 │ +0.577
```

JSON-форма (`--json` или ответ API) — `ScamReport`: `listing_id`, `score`, `risk_level`,
`base_value` (log-odds до признаков), `contributions[]` (все признаки, SHAP-вклад,
`base_value + Σ contribution = logit(P)`), `reasons[]`, `duplicate_photos[]`
(чьё фото, расстояния pHash/dHash, цена и район того объявления).

Причина попадает в отчёт, только если сигнал реально найден *и* его SHAP-вклад толкает в сторону
«мошенничества» (> 0.02). Поэтому у listing_01 (жертвы кражи фото) причина про совпадающие фото
есть, но итоговый балл низкий — остальное в объявлении чистое.

## Структура

```
src/scam_detector/
  models.py         Listing, ScamReport, FeatureContribution, PhotoMatch, RiskLevel
  images.py         hash_image (pHash+dHash), PhotoIndex, dup_stats
  text_features.py  регэкспы RU/UZ/EN, generic_description, DistrictTable, price_deviation,
                    TextIndex (Jaccard по шинглам), PhoneRegistry
  features.py       FeatureContext (справочник + корпус) → вектор из 22 признаков; load_listing
  synth.py          генератор синтетических объявлений и датасета
  synth_images.py   «фото комнат» Pillow-ом и их перезаливы (ресайз + JPEG + яркость)
  train.py          обучение CatBoost, holdout-метрики, сохранение модели и важностей
  predict.py        ScamDetector: Scam Score, ShapValues, причины; build_context
  cli.py            typer: make-dataset / train / check / batch / serve
  api.py            FastAPI: POST /check, GET /health
scripts/            make_dataset.py, make_samples.py
data/               district_prices.csv, samples/*.json + samples/images/, dataset.csv
models/             scam.cbm (в .gitignore), metrics.json, feature_importance.json
tests/              54 теста, офлайн, ~4 с
```

## Переменные окружения

См. `.env.example`; все необязательны. Файл `.env` автоматически не читается — экспортируйте
переменные в shell.

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `SCAM_MODEL_PATH` | `models/scam.cbm` | модель для check/batch/API |
| `SCAM_DISTRICTS_CSV` | `data/district_prices.csv` | справочник медиан по районам |
| `SCAM_CORPUS_DIR` | `data/samples` | корпус для дубликатов в API |
| `SCAM_UZS_PER_USD` | `12700` | курс для цен в сумах |
| `SCAM_PHOTO_THRESHOLD` | `10` | порог Хэмминга для дубликатов фото |

## Честная оговорка: данные синтетические

- **Датасет сгенерирован** (`synth.py`), реальных размеченных объявлений нет. Вероятности сигналов
  у мошенников и честных объявлений заданы вручную (например, предоплата 40% против 6%, украденные
  фото 50% против 0%, 25% мошенников ставят почти рыночную цену, честные иногда пишут «срочно»
  и «пишите в Telegram», агентства переиспользуют телефоны и перезаливают свои объявления,
  2% меток перевёрнуты). Поэтому AUC 0.94 говорит о том, что модель выучила *заложенные* правила,
  а не о качестве на реальном рынке. Для продакшена нужна разметка модераторов OLX.uz / Uybor.
- **Фото в датасете — не картинки, а хэши**: для 2000 объявлений генерируются случайные 64-битные
  pHash/dHash, «украденные» фото — копии хэшей жертвы с 0–4 перевёрнутыми битами. Настоящее
  хэширование изображений проверяется на демо-картинках и в тестах (перезалив даёт расстояние
  2–8 бит, разные комнаты — от 16).
- **Справочник цен** `district_prices.csv` — правдоподобные, но выдуманные медианы $/м²/мес.
- Корпусные признаки (`phone_reuse`, дубликаты фото и текста) зависят от размера корпуса: на папке
  из 6 объявлений они слабее, чем на датасете из 2000.

## Что замокано / упрощено

- Нет парсера площадок: объявления приходят JSON-файлами или телом запроса к API.
- Нет хранилища: корпус собирается в памяти при старте (из папки с JSON), индекс фото — numpy.
- Курс UZS→USD — константа из переменной окружения.
- LLM не используется, поэтому `llm.py` из общих соглашений в этом проекте не нужен; ключи не требуются.
- Тесты не ходят в сеть; модель для тестов обучается в фикстуре во временной папке
  (600 объявлений, 150 итераций), модель из `models/` не используется.
