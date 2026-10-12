from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Тесты никогда не ходят в сеть: принудительно MockLLM, ключи стерты."""
    monkeypatch.setenv("LLM_MOCK", "1")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)


@pytest.fixture
def data_dir() -> Path:
    return DATA


@pytest.fixture
def resume_ru_path() -> Path:
    return DATA / "sample_resume.md"


@pytest.fixture
def resume_en_path() -> Path:
    return DATA / "sample_resume_en.txt"


@pytest.fixture
def vacancy_ru_path() -> Path:
    return DATA / "sample_vacancy.txt"


@pytest.fixture
def vacancy_en_path() -> Path:
    return DATA / "vacancies" / "ai_engineer_en.txt"


@pytest.fixture
def resume_ru(resume_ru_path: Path):
    from ats_tailor.parsers import load_resume

    return load_resume(resume_ru_path)


@pytest.fixture
def resume_en(resume_en_path: Path):
    from ats_tailor.parsers import load_resume

    return load_resume(resume_en_path)


@pytest.fixture
def vacancy_ru(vacancy_ru_path: Path):
    from ats_tailor.vacancy import extract_vacancy

    return extract_vacancy(vacancy_ru_path.read_text(encoding="utf-8"))


@pytest.fixture
def vacancy_en(vacancy_en_path: Path):
    from ats_tailor.vacancy import extract_vacancy

    return extract_vacancy(vacancy_en_path.read_text(encoding="utf-8"))


def make_simple_pdf(path: Path, lines: list[str]) -> Path:
    """Минимальный текстовый PDF средствами pypdf (Helvetica, только латиница)."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(612, 792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    esc = [ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for ln in lines]
    content = "BT /F1 11 Tf 14 TL 72 740 Td " + " ".join(f"({ln}) Tj T*" for ln in esc) + " ET"
    stream = DecodedStreamObject()
    stream.set_data(content.encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as fh:
        writer.write(fh)
    return path
