from __future__ import annotations

import json

from fastapi.testclient import TestClient

from scam_detector.api import create_app

from .conftest import SAMPLES


def _payload(name: str) -> dict:
    data = json.loads((SAMPLES / name).read_text(encoding="utf-8"))
    data["photos"] = [str((SAMPLES / p).resolve()) for p in data["photos"]]
    return data


def test_check_endpoint(detector, samples_context):
    client = TestClient(create_app(detector=detector, context=samples_context))
    assert client.get("/health").json() == {"status": "ok"}
    r = client.post("/check", json=_payload("listing_02.json"))
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["score"] <= 100
    assert body["risk_level"] in {"low", "medium", "high", "critical"}
    assert body["reasons"]


def test_check_new_listing_not_in_corpus(detector, samples_context):
    client = TestClient(create_app(detector=detector, context=samples_context))
    payload = _payload("listing_02.json") | {"id": "new_1"}
    body = client.post("/check", json=payload).json()
    # новое объявление с тем же фото находит дубликат в корпусе
    assert "listing_01" in {m["other_listing_id"] for m in body["duplicate_photos"]}


def test_validation_and_missing_model(tmp_path, samples_context):
    client = TestClient(create_app(model_path=tmp_path / "none.cbm", context=samples_context))
    assert client.post("/check", json={"id": "x"}).status_code == 422
    assert client.post("/check", json=_payload("listing_01.json")).status_code == 503
