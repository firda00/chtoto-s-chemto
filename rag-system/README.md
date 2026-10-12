# rag-system — Naive RAG и Modular RAG

Две архитектуры retrieval-augmented generation на одной кодовой базе:

1. **Naive RAG** — чанкинг → эмбеддинги → косинусный top-k → контекст в LLM → ответ.
2. **Modular RAG** — препроцессинг и переписывание запроса, роутинг (коллекция + инструменты),
   гибридный поиск (BM25 + вектор, слияние Reciprocal Rank Fusion), реранкинг (cross-encoder
   или лексический), tool calling (`search_docs`, `calculator`, `get_date` в стиле OpenAI function
   calling) и генерация ответа с цитатами `[n]`.

Хранилище — за одним протоколом `VectorStore`: **Postgres + pgvector** (psycopg3, HNSW/IVFFlat)
или **InMemoryStore** на numpy (по умолчанию, работает без Docker).
База знаний для демо — синтетические документы вымышленного коворкинга *Orbita Hub* (RU + EN).

## Архитектура

### Naive RAG

```
 ingest:  data/docs/*.md ─► MarkdownChunker / FixedSizeChunker ─► Embedder ─► VectorStore
                                                                                  │
 query:   question ─► Embedder.embed_query ─► VectorStore.search (cosine top-k) ◄─┘
                                                   │
                                                   ▼
                           prompt: <context>[1]..[k]</context> + <question>
                                                   │
                                                   ▼
                                       LLM.complete ─► answer + citations
```

### Modular RAG

```
 question
   │ 1. preprocess          normalize_query (пробелы, кавычки)
   │ 2. rewrite             LLM → короткий поисковый запрос (fallback: исходный)
   │ 3. route               LLMRouter → JSON {collection, tools}  (fallback: KeywordRouter)
   ▼
 ┌──────────────── hybrid retrieval ────────────────────────────────────────┐
 │  для {исходный, переписанный} запросов:                                  │
 │     vector top-N (все коллекции)     BM25 top-N (все коллекции)          │
 │   + vector top-N (выбранная колл.)   BM25 top-N (выбранная колл.)  ×0.5  │
 │                           │                                              │
 │ 4. Reciprocal Rank Fusion  score = Σ w / (60 + rank)                     │
 │ 5. rerank → top-k          FastEmbed cross-encoder | LexicalReranker     │
 └──────────────────────────────────────────────────────────────────────────┘
   │
   ▼
 6. generation + tool loop (≤ 4 шагов)
      LLM.chat(messages, tools=[search_docs, + выбранные роутером calculator/get_date])
        ├─ tool_calls → выполнить → role=tool → снова LLM
        └─ нет tool_calls → финальный ответ
   │
   ▼
 RAGAnswer { answer, citations[n], rewritten_query, route, tool_calls }
```

Роутинг «мягкий»: выбранная коллекция получает бонус в RRF, а не жёсткий фильтр —
ошибка роутера не отрезает правильный документ.

### Модули (`src/rag_system/`)

| Модуль | Что внутри |
|---|---|
| `chunking.py` | загрузка `.md/.txt` + frontmatter, `FixedSizeChunker` (окно с перекрытием), `MarkdownChunker` (по заголовкам, путь заголовков в metadata) |
| `embeddings.py` | протокол `Embedder`, `FastEmbedEmbedder` (ONNX, ленивый), `HashingEmbedder` (numpy, детерминированный) |
| `store.py` | протокол `VectorStore`, `InMemoryStore` (+save/load), `PgVectorStore` (миграция `init()`, upsert, cosine search) |
| `bm25.py` | `BM25Index` на `rank-bm25` |
| `fusion.py` | `reciprocal_rank_fusion` с весами |
| `rerank.py` | `FastEmbedReranker` (cross-encoder), `LexicalReranker`, авто-fallback |
| `router.py` | `LLMRouter` (Pydantic-схема ответа), `KeywordRouter` |
| `tools.py` | реестр инструментов, безопасный AST-калькулятор, `get_date`, `search_docs`, `run_tool_loop` |
| `naive.py`, `modular.py` | два пайплайна |
| `llm.py` | `complete(prompt, system, json_schema)` + `chat(messages, tools)`; `OpenAILLM`, `MockLLM` |
| `factory.py` | сборка из env (`Runtime`), ingest, eval |
| `cli.py`, `api.py` | Typer CLI и FastAPI |

## Быстрый старт

```bash
cd rag-system
uv sync                                         # зависимости + dev (pytest, ruff)
uv run rag ingest data/docs                     # индекс в .rag_index/ (InMemoryStore)
uv run rag ask "Сколько стоит гостевой день?" --mode naive
uv run rag ask "Сколько будет 150 000 * 3?" --mode modular
uv run rag eval                                 # hit-rate@k и answer-contains для обоих режимов
uv run pytest -q
uv run ruff check .
```

Без `LLM_API_KEY` LLM автоматически работает в mock-режиме; явно — флаг `--mock`
(`uv run rag --mock ask ...`) или `LLM_PROVIDER=mock`.

Пример вывода (mock):

```
[modular | llm=mock | embedder=hashing | reranker=lexical]
rewritten: Сколько будет 150 000 * 3?
route: collection=all tools=['calculator'] (keyword-fallback)
tool: calculator({"expression": "150 000 * 3"}) → 450000

Ответ: calculator: 450000 ...
```

Другие команды:

```bash
uv run rag ingest data/docs --chunker fixed --chunk-size 400 --overlap 60
uv run rag ask "What is the Wi-Fi speed?" --json      # полный RAGAnswer в JSON
uv run rag eval --mode modular --k 3 --details
uv run rag serve --port 8000                          # POST /ask, POST /ingest, GET /health
uv run python scripts/generate_data.py                # пересоздать data/docs и data/eval.jsonl
```

HTTP API:

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "Можно ли прийти с собакой?", "mode": "modular"}'
```

## Переключение компонентов

### pgvector

```bash
docker compose up -d                          # pgvector/pgvector:pg16 на localhost:5432 (PG_PORT=...)
export RAG_STORE=pgvector
export PG_DSN=postgresql://rag:rag@localhost:5432/rag
uv run rag init-db                            # CREATE EXTENSION vector, таблица, HNSW-индекс
uv run rag ingest data/docs
uv run rag ask "Сколько стоит Flex?"
PG_DSN=... uv run pytest tests/test_pgvector.py   # интеграционный тест (без PG_DSN — skip)
```

`PG_INDEX=ivfflat` — IVFFlat вместо HNSW. Таблица: `rag_chunks(id, doc_id, collection, text,
metadata jsonb, embedding vector(dim))`, сходство `1 - (embedding <=> query)`.
BM25 для Modular RAG строится в памяти по `all_chunks()` — для портфолио-объёмов этого достаточно
(в проде заменяется на Postgres full-text / `tsvector`).

### fastembed (локальные ONNX-эмбеддинги и cross-encoder)

```bash
uv sync --extra local                 # ставит fastembed (без torch)
EMBEDDER=fastembed uv run rag ingest data/docs
uv run rag ask "..."                  # эмбеддер берётся из метаданных индекса
```

По умолчанию `BAAI/bge-small-en-v1.5` (англоязычная). Для русских документов лучше
`EMBED_MODEL=intfloat/multilingual-e5-small`. При установленном fastembed `RERANKER=auto`
использует cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2` (модель скачивается при первом вызове;
при ошибке — откат на лексический реранкер). Смена эмбеддера требует переиндексации — CLI
предупредит о несовпадении.

### LLM: OpenRouter / Ollama / любой OpenAI-совместимый

```bash
# OpenRouter
export LLM_API_KEY=sk-or-...  LLM_MODEL=qwen/qwen3-8b
# Ollama
ollama pull qwen3:8b
export LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen3:8b LLM_PROVIDER=openai
uv run rag ask "Сколько стоит переговорная Луна в час?"
```

Для tool calling модель должна поддерживать `tools` (Qwen3, Llama 3.1+, GPT-4o-mini и т. п.).
Блоки `<think>` reasoning-моделей вырезаются. Если провайдер не поддерживает
`response_format`, роутер повторяет запрос без него и вытаскивает JSON из текста.

## Переменные окружения

См. `.env.example` (CLI читает `.env` из текущей папки):
`LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `EMBEDDER`, `EMBED_MODEL`,
`RERANKER`, `RERANK_MODEL`, `RAG_STORE`, `RAG_INDEX_DIR`, `PG_DSN`, `PG_INDEX`, `RAG_TOP_K`.

## Что замокано

- **LLM в тестах и демо** — `MockLLM`: детерминированная заглушка без сети. Отвечает по сценарию
  (очередь строк / JSON / tool_calls), а без сценария — эвристиками: экстрактивный ответ
  (предложение контекста с максимальным пересечением терминов вопроса + цитата `[n]`),
  echo при переписывании запроса, `{}` на JSON-запрос роутера (→ keyword fallback),
  вызов `calculator`/`get_date` по шаблону вопроса. Метрики `eval` в mock-режиме проверяют
  retrieval и формат ответа, а не качество генерации.
- **Эмбеддинги в тестах** — `HashingEmbedder` (feature hashing стемов и символьных 3-грамм),
  без загрузки моделей.
- **pgvector-тест** запускается только при заданном `PG_DSN`, иначе skip.
- Все тесты офлайн, ключи не нужны.

## Тесты

```
tests/
  test_chunking.py     fixed-size с перекрытием, markdown по заголовкам, frontmatter
  test_embeddings.py   детерминизм и нормировка HashingEmbedder
  test_store.py        InMemoryStore: top-k, фильтр коллекций, upsert, save/load
  test_bm25_fusion.py  BM25, порядок RRF, веса, лексический реранкер
  test_router.py       LLMRouter со сценарием MockLLM, fallback, KeywordRouter
  test_tools.py        безопасный калькулятор, цикл tool calling со сценарием MockLLM
  test_pipelines.py    naive и modular end-to-end на data/docs, eval
  test_cli_api.py      CLI (ingest/ask/eval) и FastAPI через TestClient
  test_pgvector.py     интеграция с Postgres (skip без PG_DSN)
```
