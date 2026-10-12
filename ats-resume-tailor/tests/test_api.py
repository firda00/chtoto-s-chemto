from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ats_tailor.api import create_app
from ats_tailor.llm import MockLLM


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(llm_factory=MockLLM))


@pytest.fixture
def texts(resume_ru_path: Path, vacancy_ru_path: Path) -> dict[str, str]:
    return {
        "resume_text": resume_ru_path.read_text(encoding="utf-8"),
        "vacancy_text": vacancy_ru_path.read_text(encoding="utf-8"),
    }


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "llm": "mock"}


def test_score_json(client: TestClient, texts: dict[str, str]) -> None:
    r = client.post("/score", json=texts)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["score"]["total"] <= 100
    assert body["vacancy"]["title"] == "LLM Engineer (Middle+/Senior)"
    assert "raw_text" not in body["vacancy"]
    assert {c["name"] for c in body["score"]["components"]} == {
        "keywords",
        "similarity",
        "sections",
        "experience",
        "formatting",
    }


def test_score_multipart_md(
    client: TestClient, resume_ru_path: Path, texts: dict[str, str]
) -> None:
    r = client.post(
        "/score",
        files={"resume_file": ("cv.md", resume_ru_path.read_bytes(), "text/markdown")},
        data={"vacancy_text": texts["vacancy_text"]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["score"]["missing_must_have"] == []


def test_score_multipart_docx_and_vacancy_file(client: TestClient, vacancy_ru_path: Path) -> None:
    from docx import Document

    doc = Document()
    for line in ["Иван Петров", "ivan@example.com +7 900 111-22-33", "Навыки", "Python, Docker"]:
        doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    r = client.post(
        "/score",
        files={
            "resume_file": ("cv.docx", buf.getvalue(), "application/octet-stream"),
            "vacancy_file": ("vac.txt", vacancy_ru_path.read_bytes(), "text/plain"),
        },
    )
    assert r.status_code == 200, r.text
    score = r.json()["score"]
    assert "Python" in score["matched_keywords"]
    assert "LLM" in score["missing_must_have"]


def test_tailor_json(client: TestClient, texts: dict[str, str]) -> None:
    r = client.post("/tailor", json={**texts, "platform": "hh"})
    assert r.status_code == 200, r.text
    tailored = r.json()["tailored"]
    assert tailored["platform"] == "hh"
    assert tailored["changelog"]
    assert tailored["score_after"] >= tailored["score_before"]


def test_cover_letter_endpoint(client: TestClient, texts: dict[str, str]) -> None:
    r = client.post("/cover-letter", json=texts)
    assert r.status_code == 200, r.text
    assert "Здравствуйте" in r.json()["text"]


def test_validation_errors(client: TestClient, texts: dict[str, str]) -> None:
    assert client.post("/tailor", json={**texts, "platform": "myspace"}).status_code == 422
    assert client.post("/score", json={"resume_text": "x"}).status_code == 422
    assert client.post("/score", data={"vacancy_text": "x"}).status_code == 422
    r = client.post(
        "/score",
        files={"resume_file": ("cv.exe", b"MZ", "application/octet-stream")},
        data={"vacancy_text": "Python"},
    )
    assert r.status_code == 415
