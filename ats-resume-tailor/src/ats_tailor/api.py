"""FastAPI: POST /score, POST /tailor, POST /cover-letter.

Каждый POST принимает либо JSON ``{"resume_text": ..., "vacancy_text": ...}``, либо
multipart-форму с файлом резюме (``resume_file``: pdf/docx/md/txt) и текстом вакансии
(``vacancy_text``) или файлом вакансии (``vacancy_file``).

Запуск: ``uv run uvicorn ats_tailor.api:app --reload``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.datastructures import UploadFile

from . import __version__
from .llm import LLM, get_llm
from .models import ATSScore, CoverLetter, Resume, TailoredResume, Vacancy
from .parsers import SUPPORTED_SUFFIXES, load_document_bytes, load_resume_bytes, parse_resume
from .scoring import score_resume
from .tailor import PLATFORM_RULES, cover_letter, tailor_resume
from .vacancy import extract_vacancy, is_real_llm

MAX_UPLOAD = 5 * 1024 * 1024


class TextRequest(BaseModel):
    resume_text: str = Field(..., min_length=1)
    vacancy_text: str = Field(..., min_length=1)
    platform: str = "generic"


class ScoreResponse(BaseModel):
    vacancy: Vacancy
    score: ATSScore


class TailorResponse(BaseModel):
    vacancy: Vacancy
    tailored: TailoredResume


@dataclass
class Inputs:
    resume: Resume
    vacancy: Vacancy
    platform: str


async def _read_upload(item: Any) -> tuple[bytes, str]:
    if not isinstance(item, UploadFile):
        raise HTTPException(422, "Ожидался файл")
    data = await item.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Файл больше 5 МБ")
    return data, item.filename or "upload.txt"


async def _parse_inputs(request: Request, llm: LLM) -> Inputs:
    ctype = request.headers.get("content-type", "")
    vac_llm = llm if is_real_llm(llm) else None
    if ctype.startswith("multipart/form-data") or ctype.startswith(
        "application/x-www-form-urlencoded"
    ):
        form = await request.form()
        platform = str(form.get("platform") or "generic")
        if form.get("resume_file") is not None:
            data, name = await _read_upload(form["resume_file"])
            try:
                resume = load_resume_bytes(data, name)
            except ValueError as exc:
                raise HTTPException(
                    415, f"{exc}. Поддерживаются: {sorted(SUPPORTED_SUFFIXES)}"
                ) from exc
        elif form.get("resume_text"):
            resume = parse_resume(str(form["resume_text"]))
        else:
            raise HTTPException(422, "Нужен resume_file или resume_text")
        if form.get("vacancy_file") is not None:
            data, name = await _read_upload(form["vacancy_file"])
            vac_text = load_document_bytes(data, name).text
        elif form.get("vacancy_text"):
            vac_text = str(form["vacancy_text"])
        else:
            raise HTTPException(422, "Нужен vacancy_text или vacancy_file")
    else:
        try:
            body = TextRequest.model_validate(await request.json())
        except Exception as exc:
            raise HTTPException(422, f"Некорректный JSON: {exc}") from exc
        resume = parse_resume(body.resume_text)
        vac_text = body.vacancy_text
        platform = body.platform
    if platform not in PLATFORM_RULES:
        raise HTTPException(422, f"platform должна быть одной из {sorted(PLATFORM_RULES)}")
    return Inputs(resume=resume, vacancy=extract_vacancy(vac_text, vac_llm), platform=platform)


def create_app(llm_factory: Callable[[], LLM] | None = None) -> FastAPI:
    factory = llm_factory or get_llm
    api = FastAPI(title="ATS Resume Tailor", version=__version__)

    @api.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "llm": factory().name}

    @api.post("/score", response_model=ScoreResponse)
    async def score(request: Request) -> ScoreResponse:
        inp = await _parse_inputs(request, factory())
        return ScoreResponse(vacancy=inp.vacancy, score=score_resume(inp.resume, inp.vacancy))

    @api.post("/tailor", response_model=TailorResponse)
    async def tailor(request: Request) -> TailorResponse:
        llm = factory()
        inp = await _parse_inputs(request, llm)
        tr = tailor_resume(inp.resume, inp.vacancy, llm, platform=inp.platform)
        return TailorResponse(vacancy=inp.vacancy, tailored=tr)

    @api.post("/cover-letter", response_model=CoverLetter)
    async def cover(request: Request) -> CoverLetter:
        llm = factory()
        inp = await _parse_inputs(request, llm)
        return cover_letter(inp.resume, inp.vacancy, llm, platform=inp.platform)

    return api


app = create_app()
