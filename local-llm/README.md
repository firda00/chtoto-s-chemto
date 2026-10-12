# local-llm — запуск и оптимизация локальных LLM

Инструментарий для развёртывания и бенчмарка открытых моделей (семейства **Qwen** и **Llama**)
через **Ollama** и **vLLM**: подбор квантования и параметров генерации под бюджет GPU/CPU-памяти,
оценка VRAM, генерация `Modelfile` и команд `vllm serve`, замер TTFT / tokens/sec / латентности
на любом OpenAI-совместимом endpoint и сравнение двух конфигураций бок о бок.

## Что умеет

| Команда | Что делает |
|---|---|
| `llm presets` | Каталог пресетов (Qwen2.5 7B/14B, Qwen3 8B, Llama 3.1 8B, Llama 3.2 3B) с рекомендованным квантованием и контекстом для 8/12/16/24 GB |
| `llm estimate` | Оценка VRAM: веса + KV-кэш + runtime, с проверкой «влезает ли» в типовые GPU |
| `llm recommend` | Подбор модели + квантования + контекста под VRAM и цель (`latency` / `balanced` / `quality`) |
| `llm modelfile` | Ollama `Modelfile` из пресета (`num_ctx`, `temperature`, `top_p`, `repeat_penalty`, `num_gpu`, тег с квантованием) |
| `llm vllm-cmd` | Команда `vllm serve` или docker-compose (`--quantization awq/gptq/fp8`, `--max-model-len`, `--gpu-memory-utilization`, `--tensor-parallel-size`) |
| `llm bench` | Бенчмарк TTFT / tok/s / латентности: N промптов × M повторов, таблица + JSON/CSV, A/B-сравнение |
| `llm compare` | Сравнение двух сохранённых прогонов |
| `llm ollama list/pull/show` | Нативный API Ollama (`/api/tags`, `/api/pull`, `/api/show`) |
| `llm ask` | Один запрос через единый клиент (`LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL`) |

## Архитектура

```
presets/*.yaml ──► presets.py ──► estimate_memory_gb()  ◄── QUANTS (бит/вес, качество)
   (каталог)       (pydantic-       (веса + KV-кэш + runtime)
                    валидация)              │
                         │                  ▼
                         ├──────────► tune.py: recommend(vram, target, backend)
                         │
                         ├──► backends/ollama.py: Modelfile, OllamaClient (/api/*, /v1)
                         └──► backends/vllm.py:   vllm serve / compose, VLLMClient (/v1, /health)

client.py (LLMClient, MockLLM, complete())  ◄── llm.py (единая точка входа по CONVENTIONS)

bench.py ── httpx stream SSE ──► любой OpenAI-совместимый /v1/chat/completions
   │                              (Ollama :11434/v1, vLLM :8000/v1, OpenRouter)
   └─► rich-таблицы, JSON, CSV, сравнение A/B
mock_server.py — фейковый in-process сервер (SSE с заданными TTFT/tps, виртуальные часы)
cli.py — typer: `uv run llm ...`
```

## Установка

```bash
cd local-llm
uv sync                     # зависимости + dev (pytest, ruff, respx)
cp .env.example .env        # при необходимости
uv run llm --help
```

### Ollama (самый простой путь, CPU или GPU)

- Windows / macOS: установщик с <https://ollama.com/download>; Linux: `curl -fsSL https://ollama.com/install.sh | sh`
- Или Docker: `docker compose up -d ollama` (для GPU раскомментируйте блок `deploy` в `docker-compose.yml`)

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama list
# OpenAI-совместимый endpoint: http://localhost:11434/v1
```

Полезные переменные сервера Ollama: `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`
(KV-кэш в 8 бит ≈ вдвое больше контекста в той же памяти), `OLLAMA_KEEP_ALIVE=30m`,
`OLLAMA_NUM_PARALLEL=1` для честного замера латентности.

### vLLM (максимальная пропускная способность, нужен NVIDIA GPU, Linux/WSL2)

```bash
pip install vllm        # в отдельном окружении Linux/WSL2 с CUDA
uv run llm vllm-cmd --preset qwen2.5-7b --quant awq --max-model-len 16384   # сгенерировать команду
# или Docker (нужен NVIDIA Container Toolkit):
docker compose --profile vllm up -d vllm
# OpenAI-совместимый endpoint: http://localhost:8000/v1
```

## Быстрый старт

```bash
uv run llm presets                                   # каталог + рекомендации по VRAM
uv run llm presets --backend vllm
uv run llm estimate --model qwen2.5-7b --quant q4_K_M --ctx 8192
uv run llm estimate --params 70 --quant q4_K_M --ctx 8192 --layers 80 --kv-heads 8   # своя модель
uv run llm recommend --vram 12                        # balanced, Ollama
uv run llm recommend --vram 24 --target quality --backend vllm
uv run llm modelfile --preset qwen2.5-7b --quant q4_K_M --ctx 8192 --num-gpu 99 -o Modelfile
ollama create qwen2.5-7b-q4_k_m-8k -f Modelfile
uv run llm vllm-cmd --preset qwen3-8b --quant fp8 --max-model-len 32768 --compose
uv run llm bench --base-url http://localhost:11434/v1 --model qwen2.5:7b-instruct-q4_K_M \
    --prompts data/prompts.txt --repeats 3 --json out/q4.json --csv out/q4.csv
uv run llm bench --model qwen2.5:7b-instruct-q4_K_M --model-b qwen2.5:7b-instruct-q8_0   # A/B
uv run llm compare out/q4.json out/q8.json
uv run llm bench --mock --model qwen2.5:7b-instruct-q4_K_M --model-b qwen2.5:7b-instruct-q8_0  # без сервера
```

Пример `llm estimate --model qwen2.5-7b --quant q4_K_M --ctx 8192`:

```
│ Веса (× overhead 1.10)          │ 4.73 │
│ KV-кэш (16 бит, batch 1)        │ 0.44 │
│ Runtime (CUDA-контекст, буферы) │ 0.50 │
│ Итого                           │ 5.66 │
Помещается (90% VRAM): 8 GB: ✓  12 GB: ✓  16 GB: ✓  24 GB: ✓
```

Пример `llm recommend --vram 12`:

```
Рекомендация для 12 GB · цель balanced · ollama: Qwen3 8B · q4_K_M · ctx 16384 · ~7.8 GiB из 10.8
│ 1 │ qwen3-8b    │ q4_K_M │ 16384 │  7.84 │ 0.741 │ 0.197 │ 0.797 │
│ 2 │ qwen3-8b    │ q8_0   │  8192 │ 10.54 │ 0.776 │ 0.112 │ 0.789 │
│ 3 │ qwen2.5-14b │ q4_K_M │  4096 │ 10.42 │ 0.760 │ 0.109 │ 0.762 │
```

## Оценка памяти

```
weights  = params × bits_per_weight / 8 × overhead(1.10)
kv_cache = 2 × n_layers × n_kv_heads × head_dim × ctx_len × batch × kv_bits / 8
total    = weights + kv_cache + runtime(0.5 GiB)
```

`estimate_memory_gb(params_b, quant_bits, ctx_len, n_layers=..., n_kv_heads=..., head_dim=128,
kv_bits=16, batch=1, overhead=1.10, runtime_gb=0.5)` — `quant_bits` можно передать числом
или именем (`"q4_K_M"`). Пример: Llama 3.1 8B (32 слоя, 8 KV-голов) при 8k контекста держит
ровно 1 GiB KV-кэша в fp16; у Qwen2.5 7B всего 4 KV-головы — KV вдвое дешевле на токен.
Рекомендации считаются с запасом 10% VRAM (`--headroom`).

## Шпаргалка по квантованию

| Формат | Бэкенд | Бит/вес | 7–8B, веса | Качество vs fp16 | Когда брать |
|---|---|---|---|---|---|
| `q4_K_M` | Ollama / llama.cpp | ~4.85 | ~4.7 GiB | ~95%, потери малозаметны в чате | Дефолт: 8 GB GPU, ноутбуки, CPU |
| `q5_K_M` | Ollama | ~5.7 | ~5.5 GiB | ~97–98% | Если есть лишние 1–2 GB — заметно лучше в коде/математике |
| `q8_0` | Ollama | ~8.5 | ~8.3 GiB | ~99.5%, практически без потерь | 12–16 GB, когда важна точность |
| `fp16` / `bf16` | оба | 16 | ~15.6 GiB | эталон | 24 GB+, эталон для замеров качества |
| `awq` (int4) | vLLM | ~4.25 | ~4.1 GiB | ~95%, лучше GPTQ на инструкциях | vLLM на 8–12 GB, быстрые Marlin-ядра |
| `gptq` (int4) | vLLM | ~4.25 | ~4.1 GiB | ~94–95% | Когда нет AWQ-чекпоинта |
| `fp8` (W8A8) | vLLM | 8 | ~8 GiB | ~99% | RTX 40xx / L4 / H100 (Ada/Hopper): x2 к fp16 по памяти и скорости |

Практика:
- Декодинг упирается в пропускную способность памяти: tok/s ≈ bandwidth / размер весов.
  Поэтому 4-битные модели генерируют быстрее 8-битных примерно пропорционально битам.
- Длинный контекст стоит дорого: KV-кэш растёт линейно с `num_ctx` / `--max-model-len`.
  Не ставьте 32k «на всякий случай» — это гигабайты VRAM и выше TTFT на длинных промптах.
- KV-кэш в 8 бит (`OLLAMA_KV_CACHE_TYPE=q8_0`, vLLM `--kv-cache-dtype fp8`) ≈ x2 контекста почти без потерь.
- `num_gpu` в Ollama — число слоёв на GPU; если модель не влезает, частичный offload работает,
  но скорость падает в разы.
- vLLM резервирует `--gpu-memory-utilization` от всей VRAM и отдаёт остаток под KV-кэш батча —
  это даёт высокую пропускную способность при многих параллельных запросах.

Значения качества — ориентировочные (по публичным замерам perplexity/бенчмаркам), индекс
качества моделей в YAML — эвристика для ранжирования, а не результат бенчмарка.

## Бенчмарк

- **TTFT** — время до первого непустого токена (через стриминг SSE).
- **decode tok/s** — `(N − 1) / (t_последнего − t_первого токена)`: чистая скорость генерации.
- **e2e tok/s** — `N / полное время`; **латентность** p50/p95.
- `N` — из `usage.completion_tokens` (`stream_options.include_usage`), иначе по числу чанков.
- Запросы последовательные (batch = 1), первые `--warmup` запросов не учитываются
  (в Ollama первый запрос загружает модель в VRAM).

Пример вывода (`--mock`, симуляция q4_K_M vs q8_0, 10 промптов × 3):

```
┌────────────────────────────┬──────────┬────────┬──────────────┬──────────────┬──────────────┬───────────┬────────┬────────┐
│ Конфиг                     │ Запросов │ Ошибок │ TTFT p50, мс │ TTFT p95, мс │ decode tok/s │ e2e tok/s │ p50, с │ p95, с │
├────────────────────────────┼──────────┼────────┼──────────────┼──────────────┼──────────────┼───────────┼────────┼────────┤
│ qwen2.5:7b-instruct-q4_K_M │       30 │      0 │        180.0 │        180.0 │         45.0 │      41.9 │   2.37 │   2.78 │
│ qwen2.5:7b-instruct-q8_0   │       30 │      0 │        238.3 │        238.3 │         25.7 │      24.4 │   4.07 │   4.79 │
└────────────────────────────┴──────────┴────────┴──────────────┴──────────────┴──────────────┴───────────┴────────┴────────┘
│ decode tok/s (mean)  │  45.00 │  25.68 │  -42.9% │ qwen2.5:7b-instruct-q4_K_M │
│ Латентность p50, с   │   2.37 │   4.07 │  +72.0% │ qwen2.5:7b-instruct-q4_K_M │
```

> Это **симулированные** числа фейкового сервера (tps ∝ 1/бит-на-вес), а не замер реального GPU.
> Реальные значения получите той же командой без `--mock` против запущенного Ollama/vLLM.

## Переменные окружения

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` (у `bench` — `http://localhost:11434/v1`) | OpenAI-совместимый endpoint |
| `LLM_API_KEY` | — | Ключ (для Ollama/vLLM — любая строка) |
| `LLM_MODEL` | `qwen/qwen3-8b` | Модель для `ask`/`bench` |
| `LLM_MOCK` | `0` | `1` — `MockLLM` вместо сети |
| `OLLAMA_HOST` | `http://localhost:11434` | Нативный API Ollama |
| `LOCAL_LLM_PRESETS_DIR` | `./presets` | Свой каталог пресетов |
| `HUGGING_FACE_HUB_TOKEN` | — | Для gated-моделей Llama в vLLM |

## Добавить свой пресет

Положите `presets/<id>.yaml` (имя файла = `id`): параметры, архитектура (`n_layers`, `n_kv_heads`,
`head_dim`), префикс тега Ollama, HF-репозитории vLLM, параметры генерации и рекомендации по тирам
8/12/16/24 GB. Pydantic валидирует совместимость квантований с бэкендом и `ctx ≤ max_ctx`,
а тест `test_recommendations_cover_all_tiers_and_fit_budget` проверяет, что каждая рекомендация
действительно помещается в 90% VRAM по формуле оценки.

## Тесты

```bash
uv run pytest -q
uv run ruff check .
```

Покрыто: математика оценки памяти, загрузка/валидация пресетов, логика `recommend` для разных
бюджетов и целей, рендер Modelfile, команды и compose vLLM, нативный API Ollama (respx),
бенчмарк на мок-стриминге с детерминированными TTFT/tps (виртуальные часы), JSON/CSV,
A/B-сравнение, CLI-smoke через `typer.testing.CliRunner`. Все тесты офлайн.

## Что замокано / не проверено

- **Ollama и vLLM на машине разработки не установлены.** Все тесты и `--mock`-режим работают
  против фейкового in-process сервера (`mock_server.py`, httpx `MockTransport`) и `respx`.
  Реальный запуск `ollama create`, `vllm serve` и docker-compose не проверялся.
- Числа в `--mock` — симуляция (TTFT/tps задаются параметрами, q8_0 медленнее q4_K_M
  пропорционально битам), не измерения железа.
- Теги Ollama и HF-репозитории квантованных чекпоинтов взяты из публичных каталогов —
  перед загрузкой проверьте их наличие (особенно сторонние AWQ/GPTQ/FP8 для Llama).
- Формула памяти — оценка (±10–15%): реальное потребление зависит от версии бэкенда,
  flash attention, размера батча и CUDA-графов.
- Индекс качества моделей и квантований — эвристика для ранжирования.
- `MockLLM` — детерминированная заглушка для `complete()`; сеть в тестах не используется.
