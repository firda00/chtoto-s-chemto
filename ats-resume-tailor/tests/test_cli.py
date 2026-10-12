from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ats_tailor.cli import app

runner = CliRunner()


def test_score_cli(resume_ru_path: Path, vacancy_ru_path: Path) -> None:
    r = runner.invoke(
        app, ["score", "-r", str(resume_ru_path), "-v", str(vacancy_ru_path), "--mock"]
    )
    assert r.exit_code == 0, r.output
    assert "ATS Score" in r.output
    assert "Рекомендации" in r.output


def test_score_cli_json(resume_en_path: Path, vacancy_en_path: Path) -> None:
    r = runner.invoke(
        app, ["score", "-r", str(resume_en_path), "-v", str(vacancy_en_path), "--json", "--mock"]
    )
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert 0 <= data["total"] <= 100


def test_vacancy_cli(vacancy_en_path: Path) -> None:
    r = runner.invoke(app, ["vacancy", "-v", str(vacancy_en_path)])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["min_years"] == 4


def test_tailor_cli(tmp_path: Path, resume_ru_path: Path, vacancy_ru_path: Path) -> None:
    r = runner.invoke(
        app,
        [
            "tailor",
            "-r",
            str(resume_ru_path),
            "-v",
            str(vacancy_ru_path),
            "--platform",
            "linkedin",
            "--out",
            str(tmp_path),
            "--mock",
        ],
    )
    assert r.exit_code == 0, r.output
    for name in ("sample_resume_linkedin.md", "sample_resume_linkedin.docx"):
        assert (tmp_path / name).exists()
    assert "Changelog" in r.output


def test_cover_letter_cli(tmp_path: Path, resume_ru_path: Path, vacancy_ru_path: Path) -> None:
    out = tmp_path / "letter.md"
    r = runner.invoke(
        app,
        ["cover-letter", "-r", str(resume_ru_path), "-v", str(vacancy_ru_path), "-o", str(out)],
    )
    assert r.exit_code == 0, r.output
    assert out.read_text(encoding="utf-8").startswith("Здравствуйте")


def test_bad_platform(resume_ru_path: Path, vacancy_ru_path: Path, tmp_path: Path) -> None:
    r = runner.invoke(
        app,
        [
            "tailor",
            "-r",
            str(resume_ru_path),
            "-v",
            str(vacancy_ru_path),
            "-p",
            "myspace",
            "-o",
            str(tmp_path),
        ],
    )
    assert r.exit_code != 0
