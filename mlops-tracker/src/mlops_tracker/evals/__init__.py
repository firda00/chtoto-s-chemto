"""Evaluation harness: датасеты, скореры, прогон и отчёты."""

from mlops_tracker.evals.dataset import Case, Dataset, load_dataset
from mlops_tracker.evals.harness import (
    CaseResult,
    EvalResult,
    TaskOutput,
    load_eval_json,
    load_eval_result,
    run_eval,
)
from mlops_tracker.evals.scorers import (
    SCORERS,
    Score,
    ScoreContext,
    Scorer,
    build_scorer,
    build_scorers,
)

__all__ = [
    "SCORERS",
    "Case",
    "CaseResult",
    "Dataset",
    "EvalResult",
    "Score",
    "ScoreContext",
    "Scorer",
    "TaskOutput",
    "build_scorer",
    "build_scorers",
    "load_dataset",
    "load_eval_json",
    "load_eval_result",
    "run_eval",
]
