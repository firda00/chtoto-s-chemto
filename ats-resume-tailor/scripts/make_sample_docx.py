"""Собрать data/sample_resume.docx из data/sample_resume.md (для проверки загрузки DOCX).

Запуск: uv run python scripts/make_sample_docx.py
Данные синтетические: имя, контакты и компании вымышлены.
"""

from __future__ import annotations

from pathlib import Path

from ats_tailor.export import write_docx
from ats_tailor.models import TailoredResume
from ats_tailor.parsers import load_resume

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    resume = load_resume(ROOT / "data" / "sample_resume.md")
    doc = TailoredResume(
        platform="generic",
        vacancy_title="",
        language=resume.language,
        sections=resume.sections,
    )
    out = write_docx(doc, ROOT / "data" / "sample_resume.docx")
    print(f"written: {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
