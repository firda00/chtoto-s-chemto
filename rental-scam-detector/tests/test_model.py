from __future__ import annotations

import json
import math
from pathlib import Path

from scam_detector.features import FEATURE_NAMES, load_listing
from scam_detector.models import RiskLevel

from .conftest import SAMPLES


def test_training_reaches_auc(trained):
    assert trained.auc > 0.8
    assert set(trained.importances) == set(FEATURE_NAMES)
    out = Path(trained.model_path).parent
    assert json.loads((out / "metrics.json").read_text(encoding="utf-8"))["auc"] == trained.auc
    assert (out / "feature_importance.json").exists()


def test_report_score_range_reasons_and_shap(detector, samples_context):
    listing = load_listing(SAMPLES / "listing_02.json")
    rep = detector.report(listing, samples_context)
    assert 0 <= rep.score <= 100
    assert rep.risk_level == RiskLevel.from_score(rep.score)
    assert len(rep.contributions) == len(FEATURE_NAMES)
    assert rep.reasons and all(isinstance(r, str) for r in rep.reasons)
    # SHAP аддитивен: base + сумма вкладов = logit(p)
    logit = rep.base_value + sum(c.contribution for c in rep.contributions)
    assert abs(100 / (1 + math.exp(-logit)) - rep.score) < 1.0
    # фото listing_02 — перезалив фото listing_01
    assert any(m.other_listing_id == "listing_01" for m in rep.duplicate_photos)


def test_scam_samples_score_higher_than_legit(detector, samples_context):
    scores = {
        p.stem: detector.report(load_listing(p), samples_context).score
        for p in sorted(SAMPLES.glob("*.json"))
    }
    assert min(scores["listing_02"], scores["listing_04"]) > max(
        scores["listing_03"], scores["listing_05"]
    )
    assert scores["listing_02"] >= 50


def test_legit_listing_scores_low(detector, samples_context):
    rep = detector.report(load_listing(SAMPLES / "listing_05.json"), samples_context)
    assert rep.score < 50
    assert rep.reasons
