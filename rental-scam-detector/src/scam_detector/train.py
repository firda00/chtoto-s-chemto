"""Обучение CatBoostClassifier на синтетическом датасете, метрики на отложенной выборке."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .features import CAT_FEATURES, FEATURE_NAMES


@dataclass
class TrainResult:
    model_path: str
    n_train: int
    n_test: int
    scam_ratio: float
    auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float
    importances: dict[str, float]


def make_pool(df: pd.DataFrame, label: pd.Series | None = None) -> Pool:
    x = df[FEATURE_NAMES].copy()
    for c in CAT_FEATURES:
        x[c] = x[c].astype(str)
    return Pool(x, label=label, cat_features=CAT_FEATURES)


def train_model(
    df: pd.DataFrame,
    out_dir: str | Path = "models",
    iterations: int = 400,
    depth: int = 4,
    learning_rate: float = 0.05,
    test_size: float = 0.2,
    seed: int = 42,
) -> TrainResult:
    """Стратифицированный сплит 80/20, обучение, метрики, сохранение модели и важностей."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )
    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        loss_function="Logloss",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        # 12 районов → one-hot вместо CTR: на этом объёме CTR-статистики в ~40 раз медленнее
        one_hot_max_size=16,
    )
    model.fit(make_pool(train_df, train_df["label"]))

    proba = model.predict_proba(make_pool(test_df))[:, 1]
    y = test_df["label"].to_numpy()
    pred = (proba >= 0.5).astype(int)
    importances = dict(
        sorted(
            zip(FEATURE_NAMES, map(float, model.get_feature_importance()), strict=True),
            key=lambda kv: -kv[1],
        )
    )
    model_path = out_dir / "scam.cbm"
    model.save_model(str(model_path))
    result = TrainResult(
        model_path=str(model_path),
        n_train=len(train_df),
        n_test=len(test_df),
        scam_ratio=float(np.mean(df["label"])),
        auc=float(roc_auc_score(y, proba)),
        f1=float(f1_score(y, pred)),
        precision=float(precision_score(y, pred, zero_division=0)),
        recall=float(recall_score(y, pred)),
        accuracy=float(accuracy_score(y, pred)),
        importances=importances,
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                **{
                    k: v
                    for k, v in asdict(result).items()
                    if k not in {"importances", "model_path"}
                },
                "model_file": model_path.name,  # без абсолютного пути — файл можно коммитить
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out_dir / "feature_importance.json").write_text(
        json.dumps(importances, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result
