# Chess Match Analyzer & Mistake Explainer

Сервис анализа шахматных партий: читает PGN, оценивает каждую позицию движком
(Stockfish локально, Stockfish по HTTP API или встроенный оффлайн-движок), находит
неточности/ошибки/зевки, распознаёт простые тактические мотивы и генерирует понятные
объяснения через LLM (или по шаблону). На выходе — отчёт в Markdown, HTML и JSON
с диаграммами критических позиций (SVG) и ссылками на доску анализа lichess.

## Архитектура

```
PGN (файл/строка, N партий)
   │  pgn.py        — load_games / iter_positions / заголовки
   ▼
Engine (engine.py)  — StockfishEngine (UCI) | ApiEngine (HTTP) | FakeEngine (оффлайн)
   │  одна оценка на позицию: N ходов → N+1 вызовов
   ▼
analysis.py         — eval до/после, cp loss, классификация, win%, точность, ACPL
   │  motifs.py     — висящая фигура, упущенный/допущенный мат, вилка, связка, взятие
   ▼
explain.py ── llm.py (OpenAI-совместимый API | MockLLM)  /  template_explain()
   ▼
report.py (jinja2)  — report.md, report.html (inline SVG + график оценки), report.json, boards/*.svg
   ▲
cli.py (typer: analyze, summary)      api.py (FastAPI: POST /analyze)
```

## Быстрый старт

```bash
cd chess-analyzer
uv sync
# оффлайн, без Stockfish и без ключей:
uv run chess analyze data/games/sample.pgn --engine fake --depth 14 --lang ru --out out/
# открыть out/report.html

uv run chess summary data/games/amateur.pgn                 # заголовки партий
uv run chess summary data/games/amateur.pgn --engine fake   # + точность и ACPL

uv run pytest -q
uv run ruff check .
```

Параметры `chess analyze`:

| Опция | Значение |
|---|---|
| `--engine` | `fake` \| `stockfish` \| `api` \| `auto` (Stockfish, если найден, иначе fake) |
| `--depth` | глубина анализа (Stockfish; для API — максимум 15) |
| `--fake-depth` | глубина перебора FakeEngine: 2 (быстро, по умолчанию) или 4 (точнее, ~1 с/позицию) |
| `--lang` | язык объяснений: `ru` \| `en` |
| `--llm` | `openai` \| `mock` \| `none` \| `auto` (openai при наличии `LLM_API_KEY`, иначе шаблон) |
| `--min-class` | с какого класса объяснять ходы (`inaccuracy`, `mistake` — по умолчанию, `blunder`) |
| `--out` | папка отчёта |

API:

```bash
uv run uvicorn chess_analyzer.api:app --reload
curl -X POST localhost:8000/analyze -H "Content-Type: application/json" \
     -d '{"pgn": "1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0", "engine": "fake", "lang": "en"}'
```

Ответ — JSON со списком `games` (заголовки, статистика игроков, все ходы с оценками,
классом, лучшей линией, мотивами и объяснением). `GET /health` — проверка живости.

## Stockfish

### Windows
1. Скачать архив с https://stockfishchess.org/download/ (обычно подходит сборка `avx2`).
2. Распаковать, например, в `C:\tools\stockfish\`.
3. Указать путь (PowerShell):
   ```powershell
   $env:STOCKFISH_PATH = "C:\tools\stockfish\stockfish-windows-x86-64-avx2.exe"
   ```
   или добавить папку в `PATH` и переименовать бинарник в `stockfish.exe`.

### Linux
```bash
sudo apt install stockfish        # бинарник: /usr/games/stockfish
export STOCKFISH_PATH=/usr/games/stockfish
```

Проверка: `uv run chess analyze data/games/sample.pgn --engine stockfish --depth 16`.
Тест с настоящим Stockfish (`tests/test_engine.py::test_real_stockfish_finds_mate`)
запускается только если задан `STOCKFISH_PATH`, иначе пропускается.

### Режим API (`--engine api`)
`ApiEngine` обращается к HTTP-сервису в стиле [stockfish.online](https://stockfish.online)
(API v2): `GET <STOCKFISH_API_URL>?fen=<FEN>&depth=<N>` →
`{"success": true, "evaluation": 0.36, "mate": null, "bestmove": "bestmove e2e4 ponder e7e5", "continuation": "e2e4 e7e5 ..."}`.
`evaluation` — в пешках с точки зрения белых, `mate` — со знаком. URL по умолчанию —
публичный `https://stockfish.online/api/s/v2.php` (глубина ≤ 15, есть лимиты запросов);
можно поднять свой совместимый сервис и указать `STOCKFISH_API_URL`.

## Переменные окружения

См. `.env.example` (код читает переменные из окружения; `.env` автоматически не подгружается).

| Переменная | Назначение | По умолчанию |
|---|---|---|
| `STOCKFISH_PATH` | путь к Stockfish | поиск в `PATH` |
| `STOCKFISH_API_URL` | URL Stockfish API | `https://stockfish.online/api/s/v2.php` |
| `LLM_BASE_URL` | OpenAI-совместимый endpoint | `https://openrouter.ai/api/v1` |
| `LLM_API_KEY` | ключ LLM | — (без него объяснения по шаблону) |
| `LLM_MODEL` | модель | `qwen/qwen3-8b` |

## Формулы

Все оценки берутся **с точки зрения ходившего** и обрезаются до ±1000 сп (мат = ±1000).

* **Потеря в сантипешках:** `cp_loss = max(0, eval_before − eval_after)`, где
  `eval_before` — оценка позиции до хода (т.е. при лучшем ходе движка),
  `eval_after` — оценка позиции после сыгранного хода.
* **ACPL** — средний `cp_loss` игрока за партию.
* **Win%** (как на lichess): `win = 50 + 50 · (2 / (1 + e^(−0.00368208 · cp)) − 1)`.
* **Точность хода** (lichess): если win% не упал — 100, иначе
  `acc = 103.1668 · e^(−0.04354 · (win_before − win_after)) − 3.1669 + 1`, обрезано до [0, 100].
* **Точность за партию** (lichess `gameAccuracy`): ряд win% белых по всем позициям режется
  на окна размера `clamp(N/10, 2, 8)`, вес хода = стандартное отклонение окна (0.5…12);
  итог = (взвешенное среднее + гармоническое среднее точностей ходов игрока) / 2.

## Классификация ходов

| Класс | Условие |
|---|---|
| `best` | сыгран лучший ход движка или `cp_loss = 0` |
| `excellent` | `cp_loss ≤ 20` |
| `good` | `cp_loss ≤ 50` |
| `inaccuracy` | `50 < cp_loss ≤ 100` |
| `mistake` | `100 < cp_loss ≤ 300` |
| `blunder` | `cp_loss > 300` |

Учёт матов (применяется до порогов):
1. ход **допустил форсированный мат** (а до хода мата против ходившего не было) → `blunder`;
2. ход **упустил форсированный мат**: если оценка после хода < +3 → `blunder`, < +7 → `mistake`,
   иначе не лучше `inaccuracy`;
3. мат стал длиннее, но остался / мат поставлен / «уже матуют» — классифицируется по cp loss
   (обе оценки = ±1000, потеря ≈ 0).

## Тактические мотивы (эвристики, `motifs.py`)

Проверяются только для неточностей, ошибок и зевков; основаны на картах атак python-chess:

* **hanging_piece** — после хода фигура (≥ коня) атакована и не защищена или атакована более
  дешёвой фигурой (связки атакующих и промежуточные ходы не учитываются);
* **missed_mate** — у ходившего был мат (по движку, либо перебором найден мат в 1), сыгран другой ход;
* **allowed_mate** — после хода у соперника появился мат (по движку или мат в 1);
* **missed_fork / allowed_fork** — лучший ход (наш / ответ соперника) — вилка: фигура в
  безопасности и бьёт ≥ 2 целей (король, более ценная фигура или незащищённая фигура ≥ коня);
* **missed_pin / allowed_pin** — лучший ход создаёт новую абсолютную связку (к королю);
* **missed_capture** — лучший ход забирает фигуру ≥ коня.

Мотивы передаются в промпт LLM и в шаблонное объяснение.

## Объяснения

`explain.py` строит промпт с FEN, сыгранным и лучшим ходом, линией (PV в SAN), оценками
до/после, сдвигом оценки и мотивами; системный промпт просит 2–3 предложения на русском
или английском (`--lang`). Вызов идёт через `llm.complete(prompt, system=None, json_schema=None)`.
Если LLM не настроен, упал или вернул пустой ответ — используется `template_explain()`.

## Пример отчёта

Фрагмент `out/report.md` (оффлайн, `--engine fake --fake-depth 4`), партия Легаля (1750):

```
#### 5... Bxd1?? — грубая ошибка (зевок)
- FEN: `rn1qkbnr/ppp2p1p/3p2p1/4N3/2B1P1b1/2N5/PPPP1PPP/R1BQK2R b KQkq - 0 5`
- Сыграно: **Bxd1**, Лучше: **dxe5**
- Оценка (для ходившего): -2.30 → #-2 (Потеря: 770 cp)
- Мотивы: сопернику разрешён мат в 2 (Bxf7+); слон на d1 остаётся под боем без
  достаточной защиты; упущено взятие: конь на e5 (dxe5)
- [Открыть на lichess](https://lichess.org/analysis/standard/rn1qkbnr/ppp2p1p/...)
```

HTML-версия содержит сводную таблицу (точность, ACPL, число ходов каждого класса),
график оценки и карточку на каждый критический ход: SVG-доска (красная стрелка — сыгранный
ход, зелёная — лучший), FEN, линия, мотивы, объяснение и ссылка на lichess.

## Демо-данные

`data/games/` генерируется скриптом `uv run python scripts/make_games.py` (ходы проверяются
python-chess на легальность, финальные маты — на `is_checkmate()`):

* `sample.pgn` — две миниатюры: детский мат (3...Nf6?? 4.Qxf7#) и мат Легаля (5...Bxd1??);
* `amateur.pgn` — синтетическая клубная партия (48 полуходов) с зевками 13...Nh5??, 17.Qg4?? и др.;
* `missed_mate.pgn` — белые упускают 4.Qxf7#, чёрные позже упускают мат и в итоге матуют 13...Qe1#.

## Что замокано / ограничения

* **FakeEngine** — не Stockfish: негамакс с альфа-бета на 2 (или 4) полухода, оценка =
  материал + бонус за центр, ходы перебираются детерминированно (взятия первыми, затем по UCI).
  Видит мат в 1 (depth 2) / мат в 2 (depth 4) и прямые потери материала, но не глубокую
  тактику — например, на depth 2 жертву 5.Nxe5 в партии Легаля он считает зевком.
  В отчёте показывается запрошенная `--depth`; FakeEngine использует `--fake-depth`.
* **MockLLM** — детерминированная заглушка, собирает ответ из полей промпта; используется в тестах.
  Без `LLM_API_KEY` CLI/API по умолчанию дают шаблонные объяснения.
* **ApiEngine** в тестах проверяется через `httpx.MockTransport` — в сеть тесты не ходят.
* Тест с настоящим Stockfish пропускается без `STOCKFISH_PATH`.
* Мотивы — эвристики (см. выше), возможны ложные срабатывания.

## Тесты

```bash
uv run pytest -q
```

Покрыто: загрузка PGN (файл/строка, несколько партий, ошибки), детерминизм FakeEngine,
парсинг ответа Stockfish API, cp loss и пороги классификации с учётом матов, формулы
точности на известных значениях, детекторы мотивов на специально составленных FEN
(висящая фигура, упущенный/допущенный мат в 1, вилка, связка), объяснения через MockLLM
и шаблон (включая падение LLM), рендер Markdown/HTML (секции, SVG), FastAPI `/analyze`,
smoke-тест CLI с fake-движком.
