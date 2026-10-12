# Eval-отчёт: mock-agent

- **Ран:** `093716c9776e`  
- **Task:** `examples.tasks:mock_agent`  
- **Датасет:** `qa` (hash `39cf53cf863b`, 12 кейсов)  
- **Скореры:** contains, llm_judge, cost, latency  
- **Создан:** 2026-10-12T18:29:07+00:00, длительность 0.01 с

## Сводка

| Метрика | Значение |
|---|---|
| pass_rate | 83.3% |
| n_passed | 10 |
| n_cases | 12 |
| error_rate | 0.0% |
| latency_ms_p50 | 2.39 |
| latency_ms_p95 | 3.18 |
| cost_usd_total | $0.000184 |
| tokens_prompt | 1544 |
| tokens_completion | 536 |

## Скореры

| Скорер | Средняя оценка | Pass rate |
|---|---|---|
| contains | 0.833 | 83.3% |
| llm_judge | 0.833 | 83.3% |
| cost | 1.000 | 100.0% |
| latency | 1.000 | 100.0% |

## Кейсы

| Кейс | Статус | contains | llm_judge | cost | latency | Латентность, мс | Ответ |
|---|---|---|---|---|---|---|---|
| qa-capital-fr | PASS | 1.00 | 1.00 | 1.00 | 1.00 | 1.4 | The capital of France is Paris. |
| qa-capital-uz | PASS | 1.00 | 1.00 | 1.00 | 1.00 | 2.6 | Столица Узбекистана — Ташкент. |
| qa-math-mul | PASS | 1.00 | 1.00 | 1.00 | 1.00 | 1.9 | Ответ: 51 |
| qa-math-precedence | PASS | 1.00 | 1.00 | 1.00 | 1.00 | 3.0 | Answer: 6 |
| qa-tolstoy | PASS | 1.00 | 1.00 | 1.00 | 1.00 | 2.4 | «Войну и мир» написал Лев Толстой. |
| qa-einstein | PASS | 1.00 | 1.00 | 1.00 | 1.00 | 2.1 | The theory of relativity was developed by Albert Einstein. |
| qa-project-lang | PASS | 1.00 | 1.00 | 1.00 | 1.00 | 2.2 | Этот проект написан на Python 3.12. |
| qa-rag | PASS | 1.00 | 1.00 | 1.00 | 1.00 | 2.3 | RAG stands for Retrieval-Augmented Generation. |
| qa-planets | PASS | 1.00 | 1.00 | 1.00 | 1.00 | 2.4 | В Солнечной системе 8 планет. |
| qa-boiling | PASS | 1.00 | 1.00 | 1.00 | 1.00 | 2.7 | Water boils at 100 °C at sea level. |
| qa-mlops | FAIL | 0.00 | 0.00 | 1.00 | 1.00 | 3.1 | Не знаю. |
| qa-nile | FAIL | 0.00 | 0.00 | 1.00 | 1.00 | 3.2 | Не знаю. |

## Провалы — детали

### qa-mlops
- **Вход:** Что такое MLOps?
- **Ожидалось:** ["эксплуатац", "машинного обучения"]
- **Получено:** Не знаю.
- `contains` = 0.00: {"found": [], "missing": ["эксплуатац", "машинного обучения"]}
- `llm_judge` = 0.00: {"judge_score": 1, "reason": "mock: совпало 0% слов эталона", "judge_model": "mock-llm"}

### qa-nile
- **Вход:** Какая самая длинная река в мире?
- **Ожидалось:** Нил
- **Получено:** Не знаю.
- `contains` = 0.00: {"found": [], "missing": ["Нил"]}
- `llm_judge` = 0.00: {"judge_score": 1, "reason": "mock: совпало 0% слов эталона", "judge_model": "mock-llm"}


## Сравнение: `baseline` → `mock-agent`

Базовый ран `798826923749`, новый ран `093716c9776e`. Регрессий: **1**, исправлено: **1**.

| Метрика | База | Новый | Δ |
|---|---|---|---|
| pass_rate | 83.3% | 83.3% | +0.0% |
| n_passed | 10 | 10 | +0 |
| n_cases | 12 | 12 | +0 |
| error_rate | 0.0% | 0.0% | +0.0% |
| latency_ms_p50 | 0.06 | 2.39 | +2.337 (хуже) |
| latency_ms_p95 | 0.27 | 3.18 | +2.912 (хуже) |
| cost_usd_total | — | $0.000184 | — |
| tokens_prompt | 0 | 1544 | +1544 (хуже) |
| tokens_completion | 0 | 536 | +536 (хуже) |

| Кейс | Изменение | Было | Стало |
|---|---|---|---|
| qa-math-precedence | исправлено | Ответ: 8 | Answer: 6 |
| qa-mlops | РЕГРЕССИЯ | MLOps — практики эксплуатации моделей машинного обучения в … | Не знаю. |
