from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from examples.tasks import answer_question, extract_contact, mock_agent
from mlops_tracker.evals.dataset import Dataset, load_dataset
from mlops_tracker.evals.harness import TaskOutput, load_eval_json, load_eval_result, run_eval
from mlops_tracker.evals.scorers import Contains, ExactMatch, LLMJudge
from mlops_tracker.llm import MockLLM
from mlops_tracker.tracker import Tracker

DATA = Path(__file__).resolve().parents[1] / "data" / "evals"

ANSWERS = {"2+2": "4", "столица Франции": "Париж", "hello": "мир"}


def lookup(question: str) -> str:
    return ANSWERS[question]


def test_end_to_end_with_mock_judge(tracker: Tracker, small_dataset: Dataset, tmp_path: Path):
    judge_llm = MockLLM()
    result = run_eval(
        lookup,
        small_dataset,
        [ExactMatch(), LLMJudge(llm=judge_llm, threshold=4)],
        concurrency=3,
        tracker=tracker,
        params={"variant": "A"},
        report_dir=tmp_path / "reports",
    )
    assert [c.id for c in result.cases] == ["a", "b", "c"]  # порядок сохранён
    assert [c.passed for c in result.cases] == [True, True, False]
    m = result.metrics
    assert m["n_cases"] == 3 and m["n_passed"] == 2
    assert m["pass_rate"] == 2 / 3
    assert m["exact_match/pass_rate"] == 2 / 3
    assert m["llm_judge/mean"] == (1 + 1 + 0) / 3
    assert len(judge_llm.calls) == 3

    # всё залогировано в трекер
    run = tracker.store.get_run(result.run_id)
    assert run.status == "finished" and "eval" in run.tags
    assert run.params["variant"] == "A"
    assert run.params["dataset_hash"] == small_dataset.hash
    assert run.params["scorers"] == ["exact_match", "llm_judge"]
    assert run.last_metrics()["pass_rate"] == m["pass_rate"]
    assert len(run.metrics["case/passed"]) == 3

    # JSON-отчёт: в базе и файлом, round-trip
    assert load_eval_result(tracker.store, result.run_id).metrics == m
    report_file = Path(result.report_path)
    assert report_file.exists() and report_file.name == f"eval_{result.run_id}.json"
    assert load_eval_json(report_file).cases[2].output == "мир"
    assert f"eval_{result.run_id}.json" in run.artifacts


def test_task_errors_and_retries(tracker: Tracker) -> None:
    attempts: dict[str, int] = {}
    lock = threading.Lock()

    def flaky(q: str) -> str:
        with lock:
            attempts[q] = attempts.get(q, 0) + 1
            n = attempts[q]
        if q == "always-fails":
            raise RuntimeError("модель недоступна")
        if n < 2:
            raise ConnectionError("временная ошибка")
        return q

    ds = Dataset.from_records(
        [{"id": "ok", "input": "x", "expected": "x"}, {"id": "bad", "input": "always-fails"}]
    )
    result = run_eval(flaky, ds, "exact_match", retries=2, tracker=tracker)
    ok, bad = result.cases
    assert ok.passed and ok.attempts == 2
    assert not bad.passed and bad.attempts == 3 and "модель недоступна" in bad.error
    assert bad.scores["exact_match"].details["error"] == bad.error
    assert result.metrics["error_rate"] == 0.5
    assert result.metrics["retries_total"] == 3


def test_timeout(tracker: Tracker) -> None:
    def slow(q: str) -> str:
        if q == "slow":
            time.sleep(1.0)
        return q

    ds = Dataset.from_records(
        [
            {"id": "fast", "input": "fast", "expected": "fast"},
            {"id": "slow", "input": "slow", "expected": "slow"},
        ]
    )
    t0 = time.perf_counter()
    result = run_eval(slow, ds, [ExactMatch()], timeout=0.2, tracker=tracker)
    assert time.perf_counter() - t0 < 0.9
    assert result.cases[0].passed
    assert not result.cases[1].passed and "TaskTimeoutError" in result.cases[1].error


def test_concurrency_runs_in_parallel(tracker: Tracker) -> None:
    def sleepy(q: str) -> str:
        time.sleep(0.2)
        return q

    ds = Dataset.from_records([{"input": str(i), "expected": str(i)} for i in range(5)])
    t0 = time.perf_counter()
    result = run_eval(sleepy, ds, "exact_match", concurrency=5, tracker=tracker)
    assert time.perf_counter() - t0 < 0.8  # последовательно было бы ≥ 1.0 с
    assert result.metrics["pass_rate"] == 1.0


def test_async_task_task_output_and_case_access(tracker: Tracker) -> None:
    async def agent(q: str, case) -> TaskOutput:
        return TaskOutput(
            output=f"{q}-{case.metadata['suffix']}",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            cost_usd=0.001,
            metadata={"steps": 2},
        )

    ds = Dataset.from_records([{"input": "a", "expected": "a-z", "metadata": {"suffix": "z"}}])
    result = run_eval(agent, ds, "exact_match,cost:0.01,latency:5000", tracker=tracker)
    c = result.cases[0]
    assert c.passed and c.output == "a-z" and c.task_metadata == {"steps": 2}
    assert result.metrics["tokens_prompt"] == 10
    assert result.metrics["cost_usd_total"] == 0.001
    assert set(c.scores) == {"exact_match", "cost", "latency"}


def test_custom_function_scorer_and_no_tracker(tmp_path: Path) -> None:
    def polite(case, output) -> bool:
        return "пожалуйста" in output

    ds = Dataset.from_records([{"input": "x"}])
    result = run_eval(
        lambda q: "да, пожалуйста",
        ds,
        [polite],
        log_to_tracker=False,
        report_dir=tmp_path,
        run_name="local",
    )
    assert result.run_id is None and result.cases[0].scores["polite"].passed
    assert json.loads(Path(result.report_path).read_text("utf-8"))["run_name"] == "local"


def test_example_tasks_on_demo_datasets(tracker: Tracker) -> None:
    qa = load_dataset(DATA / "qa.jsonl")
    base = run_eval(answer_question, qa, [Contains()], tracker=tracker)
    agent = run_eval(mock_agent, qa, [Contains(), "cost:0.01"], tracker=tracker)
    assert base.metrics["n_passed"] == 10 and agent.metrics["n_passed"] == 10
    failed_base = {c.id for c in base.failed_cases}
    failed_agent = {c.id for c in agent.failed_cases}
    assert failed_base == {"qa-math-precedence", "qa-nile"}
    assert failed_agent == {"qa-mlops", "qa-nile"}
    assert agent.metrics["tokens_prompt"] > 0 and agent.metrics["cost_usd_total"] > 0
    trace = agent.cases[2].task_metadata["trace"]
    assert trace[0]["action"] == "calculator" and trace[-1]["action"] == "final"

    ex = load_dataset(DATA / "json_extraction.yaml")
    res = run_eval(extract_contact, ex, "json_schema,json_match", tracker=tracker)
    assert res.metrics["json_schema/pass_rate"] == 5 / 6
    assert res.metrics["json_match/pass_rate"] == 4 / 6
