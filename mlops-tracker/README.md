# mlops-tracker

Лёгкий трекер экспериментов и Evaluation Harness для LLM-приложений и агентов.
Можно думать о нём как о «мини-MLflow + мини-promptfoo» в одном пакете: хранение на stdlib
`sqlite3`, зависимости — только pydantic, typer, rich, jinja2, openai и pyyaml.

## Концепция

Проект состоит из двух слоёв.

1. **Трекер** отвечает на вопрос «что именно я запускал и что получилось». Каждый запуск (ран)
   хранит параметры, метрики по шагам, артефакты и тексты (промпты, конфиги). Для
   воспроизводимости в ран также записываются git-коммит, флаг «грязного» дерева, версия Python,
   ОС, версии всех установленных пакетов и random seed.
2. **Харнес** отвечает на вопрос «стала ли модель или агент лучше». Он прогоняет `task` по
   датасету кейсов, считает скореры (точное совпадение, подстроки, regex, JSON Schema, числа с
   допуском, LLM-as-a-judge, латентность, стоимость) и агрегирует результат. Каждый прогон
   логируется как ран трекера, поэтому два прогона можно сравнить и увидеть регрессии по кейсам.

## Архитектура

```
                   ┌────────────────────────── evals/ ───────────────────────────┐
 data/evals/*.jsonl│ dataset.py ─► harness.run_eval ─► scorers.py ─► report.py   │
 data/evals/*.yaml │  Case{id,input,  ThreadPool,      exact/contains/  Markdown │
                   │  expected,meta}  retries,timeout  regex/json/num/  HTML     │
                   │                  TaskOutput       llm_judge/       diff     │
                   │                       │           latency/cost             │
                   └───────────────────────┼─────────────────┬───────────────────┘
 examples/tasks.py ─► task(input) ─────────┘                 │ llm.py
                                           ▼                 ▼ complete(prompt, system, json_schema)
                    tracker.py  Run(ctx manager) ─► store.py ─► mlruns/mlt.db (SQLite)
                    git, env, seed snapshot          runs · params · metrics(step)
                                                     artifacts · texts (JSON-отчёт eval)
                                           ▲
                    cli.py (typer): mlt runs list|show|compare|best · eval run · report
```

| Модуль | Что делает |
|---|---|
| `src/mlops_tracker/store.py` | Схема SQLite и запросы: `list_runs`, `get_run` (по префиксу id), `compare_runs`, `best_run` |
| `src/mlops_tracker/tracker.py` | `Tracker`, `Run` (контекстный менеджер), снапшот окружения, seed |
| `src/mlops_tracker/llm.py` | `complete(prompt, system=None, json_schema=None)`, `OpenAICompatibleLLM`, `MockLLM` |
| `src/mlops_tracker/evals/dataset.py` | Загрузка кейсов из JSONL, JSON и YAML, hash датасета |
| `src/mlops_tracker/evals/scorers.py` | Встроенные скореры и реестр (`"latency:2000"` и т. п.) |
| `src/mlops_tracker/evals/harness.py` | `run_eval`: параллельность, ретраи, таймауты, агрегаты, логирование |
| `src/mlops_tracker/evals/report.py` | Markdown, HTML (`templates/report.html.j2`), diff двух ранов |
| `src/mlops_tracker/cli.py` | CLI `mlt` |

## Быстрый старт

```bash
cd mlops-tracker
uv sync
uv run pytest -q           # 42 теста, полностью офлайн
uv run ruff check .

# Бейзлайн: детерминированная функция, судья — MockLLM (ключ не нужен)
uv run mlt eval run --dataset data/evals/qa.jsonl \
    --task examples.tasks:answer_question --scorers contains,llm_judge \
    --name baseline --seed 42 --out out

# Агент (ReAct на MockLLM) + бюджет по стоимости и латентности
uv run mlt eval run -d data/evals/qa.jsonl -t examples.tasks:mock_agent \
    -s contains,llm_judge,cost:0.01,latency:2000 -n mock-agent --retries 1 --timeout 30 -o out

# Извлечение JSON с проверкой схемы
uv run mlt eval run -d data/evals/json_extraction.yaml -t examples.tasks:extract_contact \
    -s json_schema,json_match -n extract-contact -o out

uv run mlt runs list
uv run mlt runs show <run_id>               # достаточно префикса id
uv run mlt runs compare <baseline_id> <agent_id>
uv run mlt runs best --metric pass_rate
uv run mlt report <agent_id> --baseline <baseline_id> --html out/report.html --md out/report.md
```

Для CI есть флаг `--fail-under 0.9`: если `pass_rate` ниже порога, команда завершится с кодом 1.

Демо-датасеты генерируются скриптом: `uv run python scripts/make_datasets.py`.

## Трекер из Python

```python
from mlops_tracker import Tracker

tracker = Tracker("mlruns/mlt.db")
with tracker.start_run(
    "rag-v2", params={"top_k": 5, "model": "qwen3-8b"}, tags=["rag"], seed=42
) as run:
    for epoch in range(3):
        run.log_metric("recall@5", 0.7 + epoch * 0.05, step=epoch)
    run.log_artifact("config.yaml")
    run.log_text("system_prompt", "Ты — помощник ...")
# Если внутри блока возникло исключение, ран получит статус failed и сохранит traceback

best, value = tracker.store.best_run("recall@5", mode="max")
diff = tracker.store.compare_runs(run_a_id, run_b_id)
```

Шорткат без явного трекера: `from mlops_tracker import start_run` (база берётся из `MLT_DB`).

## Как написать свой task

Task — это любая функция `input -> output`. Если ей нужны поля кейса (например, `metadata`),
добавьте параметр `case`. Поддерживаются и `async`-функции. Агентам удобно возвращать
`TaskOutput`: харнес возьмёт из него токены и стоимость для скорера `cost`, а трейс покажет в
HTML-отчёте.

```python
# my_tasks.py
from mlops_tracker.evals import TaskOutput
from mlops_tracker.llm import get_llm

llm = get_llm()


def summarize(text: str, case) -> TaskOutput:
    answer = llm.complete(f"Кратко перескажи ({case.metadata.get('lang', 'ru')}):\n{text}")
    return TaskOutput(output=answer, usage=llm.last_usage, metadata={"model": llm.model})
```

```bash
uv run mlt eval run -d data/evals/my.jsonl -t my_tasks:summarize -s llm_judge,latency:3000
```

Формат кейса: `{"id": "...", "input": ..., "expected": ..., "metadata": {...}}`. В YAML/JSON
можно задать `metadata` на верхнем уровне: она подмешивается в каждый кейс. Так в
`json_extraction.yaml` задаётся одна JSON Schema на все кейсы.

## Как написать свой scorer

Подойдёт функция `(case, output)` или `(case, output, ctx)`. Возвращать можно `bool`, число
0..1 (pass при значении ≥ 0.5) или `Score`. В `ctx` лежат `latency_ms`, `usage`, `cost_usd`,
`attempts` и `error`. Исключение внутри скорера не роняет прогон: кейс просто получит провал с
текстом ошибки.

```python
from mlops_tracker.evals import Score, run_eval, load_dataset


def no_english(case, output) -> bool:
    return not any("a" <= ch.lower() <= "z" for ch in output)


class MaxWords:
    name = "max_words"

    def __init__(self, n):
        self.n = n

    def __call__(self, case, output, ctx):
        words = len(str(output).split())
        return Score(
            name=self.name,
            value=min(1, self.n / max(words, 1)),
            passed=words <= self.n,
            details={"words": words},
        )


result = run_eval(
    my_task,
    load_dataset("data/evals/qa.jsonl"),
    scorers=["contains", no_english, MaxWords(30)],
    concurrency=8,
    retries=2,
)
```

Чтобы скорер стал доступен по имени в CLI, добавьте его класс в `SCORERS` в `scorers.py`.

## Как подключить LLM-judge

`llm_judge` строит промпт из рубрики и тегов `<input>`, `<expected>` и `<output>`, а затем
запрашивает у модели JSON `{"score": 1..5, "reason": "..."}`. Итоговое значение считается как
`value = (score - 1) / 4`, кейс проходит при `score >= threshold` (по умолчанию 4, в CLI
задаётся как `llm_judge:3`). Рубрику можно переопределить для отдельного кейса через
`metadata.rubric`.

```bash
cp .env.example .env    # LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
uv run mlt eval run -d data/evals/qa.jsonl -t examples.tasks:answer_question \
    -s contains,llm_judge --llm openai     # auto: реальная модель, только если задан ключ
```

Из Python: `LLMJudge(llm=OpenAICompatibleLLM(model="..."), rubric="...", threshold=4)`. Если
провайдер не поддерживает `response_format=json_schema`, используется схема в тексте промпта и
JSON извлекается из ответа. Если оценку не удалось разобрать, берётся первая цифра 1–5.

## Переменные окружения

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-совместимый endpoint (Ollama: `http://localhost:11434/v1`) |
| `LLM_API_KEY` | пусто | Если не задан, используется `MockLLM` |
| `LLM_MODEL` | `qwen/qwen3-8b` | Модель для судьи и агента |
| `LLM_PROVIDER` | `auto` | `auto`, `mock` или `openai` |
| `MLT_DB` | `mlruns/mlt.db` | Путь к SQLite; артефакты лежат рядом, в `artifacts/<run_id>/` |
| `MLT_DISABLE_GIT` | `0` | `1` — не вызывать `git` при старте рана |

## Пример отчёта

В `out/` лежат результаты реального прогона CLI на MockLLM:

- `out/report.html`: агент против бейзлайна, самодостаточный HTML с diff и подсвеченными
  регрессиями;
- `out/report.md`: то же самое в Markdown;
- `out/json_extraction.html`: извлечение JSON с ошибками схемы в деталях кейсов;
- `out/eval_<run_id>.json`: полные JSON-отчёты.

Фрагмент `report.md`:

```
| Метрика | База | Новый | Δ |
|---|---|---|---|
| pass_rate | 83.3% | 83.3% | +0.0% |
| latency_ms_p50 | 0.06 | 2.39 | +2.337 (хуже) |
| tokens_prompt | 0 | 1544 | +1544 (хуже) |

| Кейс | Изменение | Было | Стало |
|---|---|---|---|
| qa-math-precedence | исправлено | Ответ: 8 | Answer: 6 |
| qa-mlops | РЕГРЕССИЯ | MLOps — практики эксплуатации моделей… | Не знаю. |
```

Итоговый `pass_rate` одинаковый, но под ним скрыты регрессия и исправление. Агент правильно
считает `2 + 2 * 2`, потому что использует калькулятор на AST, а у бейзлайна намеренный баг с
вычислением слева направо. Зато ретривер агента строже (ему нужно не меньше двух совпавших
ключевых слов), и вопрос про MLOps он уже не находит. Именно такие изменения и должен ловить diff
по кейсам.

## Что замокано и какие есть ограничения

- **LLM.** Тесты и демо работают на `MockLLM` и никогда не обращаются к сети. Mock-судья ставит
  оценку по доле слов эталона, найденных в ответе (1–5). Это эвристика, а не настоящая оценка
  качества. Агент `mock_agent` без ключа использует детерминированную политику `mock_policy`.
- **Стоимость.** Токены `MockLLM` считаются приблизительно (слова и знаки препинания), цены по
  умолчанию условные (`DEFAULT_PRICE_IN/OUT` в `scorers.py`).
- **OpenAICompatibleLLM.** В тестах проверяется только создание клиента; реальные вызовы API
  не тестировались.
- **Таймауты.** Задача, превысившая таймаут, помечается ошибкой, но её поток Python
  невозможно прервать, поэтому он доработает в фоне.
- **JSON Schema.** Валидатор встроенный и покрывает подмножество: `type`, `enum`, `const`,
  `required`, `properties`, `additionalProperties`, `items`, `min/maxItems`, `min/maxLength`,
  `pattern`, `minimum/maximum`. Ссылки `$ref` и комбинаторы `oneOf/anyOf` не поддерживаются.
- **git.** Ошибки git (git не установлен, нет коммитов, таймаут) не прерывают запуск: поле
  `git_commit` просто остаётся пустым.
