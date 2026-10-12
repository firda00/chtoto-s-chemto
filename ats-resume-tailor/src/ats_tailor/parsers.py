"""Извлечение текста из PDF/DOCX/TXT/MD и разбиение резюме на разделы (RU + EN)."""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

from .models import Contacts, DocumentInfo, Resume

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".markdown"}

# ------------------------------------------------------------------ extraction


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


_MD_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", re.M)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def _from_text(data: bytes, fmt: str) -> DocumentInfo:
    text = _decode(data).replace("\r\n", "\n")
    return DocumentInfo(
        text=text,
        fmt=fmt,
        tables=len(_MD_TABLE_SEP.findall(text)),
        images=len(_MD_IMAGE.findall(text)),
    )


def _from_pdf(data: bytes) -> DocumentInfo:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    texts: list[str] = []
    images = 0
    for page in reader.pages:
        texts.append(page.extract_text() or "")
        with contextlib.suppress(Exception):  # битые ресурсы страницы
            images += len(page.images)
    return DocumentInfo(
        text="\n".join(texts).strip(), fmt="pdf", images=images, pages=len(reader.pages)
    )


def _from_docx(data: bytes) -> DocumentInfo:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(io.BytesIO(data))
    lines: list[str] = []
    tables = 0
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, doc)
            text = para.text.strip()
            if not text:
                lines.append("")
                continue
            style = (para.style.name or "") if para.style is not None else ""
            lines.append(f"- {text}" if style.startswith("List") else text)
        elif tag == "tbl":
            tables += 1
            for row in Table(child, doc).rows:
                seen: list[str] = []
                for cell in row.cells:
                    t = cell.text.strip()
                    if t and t not in seen:
                        seen.append(t)
                if seen:
                    lines.append(" | ".join(seen))
    return DocumentInfo(
        text="\n".join(lines).strip(),
        fmt="docx",
        tables=tables,
        images=len(doc.inline_shapes),
    )


def load_document_bytes(data: bytes, filename: str) -> DocumentInfo:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _from_pdf(data)
    if suffix == ".docx":
        return _from_docx(data)
    if suffix in {".md", ".markdown"}:
        return _from_text(data, "md")
    if suffix in {".txt", ""}:
        return _from_text(data, "txt")
    raise ValueError(
        f"Неподдерживаемый формат: {suffix}. Поддерживаются: {sorted(SUPPORTED_SUFFIXES)}"
    )


def load_document(path: str | Path) -> DocumentInfo:
    p = Path(path)
    return load_document_bytes(p.read_bytes(), p.name)


def extract_text(path: str | Path) -> str:
    return load_document(path).text


# ------------------------------------------------------------------ sections

HEADINGS: dict[str, list[str]] = {
    "contacts": [
        "контакты",
        "контактная информация",
        "контактные данные",
        "личные данные",
        "contacts",
        "contact",
        "contact information",
        "contact info",
        "contact details",
        "personal information",
        "personal details",
    ],
    "summary": [
        "о себе",
        "обо мне",
        "краткое резюме",
        "профиль",
        "профессиональный профиль",
        "цель",
        "summary",
        "professional summary",
        "profile",
        "professional profile",
        "about",
        "about me",
        "objective",
        "career objective",
        "overview",
    ],
    "skills": [
        "навыки",
        "ключевые навыки",
        "профессиональные навыки",
        "технические навыки",
        "технологии",
        "стек",
        "стек технологий",
        "технологический стек",
        "компетенции",
        "skills",
        "key skills",
        "technical skills",
        "hard skills",
        "core skills",
        "core competencies",
        "competencies",
        "tech stack",
        "technologies",
        "skills and tools",
    ],
    "experience": [
        "опыт работы",
        "опыт",
        "профессиональный опыт",
        "трудовая деятельность",
        "места работы",
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "employment history",
        "work history",
        "career history",
    ],
    "education": [
        "образование",
        "высшее образование",
        "образование и курсы",
        "курсы",
        "курсы и сертификаты",
        "сертификаты",
        "повышение квалификации",
        "education",
        "courses",
        "certifications",
        "certificates",
        "education and training",
        "academic background",
    ],
    "projects": ["проекты", "pet-проекты", "pet projects", "projects", "personal projects"],
    "languages": ["языки", "знание языков", "иностранные языки", "languages"],
}

_HEADING_LOOKUP: dict[str, str] = {
    phrase: section for section, phrases in HEADINGS.items() for phrase in phrases
}


def _normalize_heading(line: str) -> str:
    s = line.strip()
    s = re.sub(r"^#{1,6}\s*", "", s)
    s = s.strip("*_=•-–—:| \t").strip()
    s = re.sub(r"\s+", " ", s).lower()
    return s.rstrip(":").strip()


def parse_heading(line: str) -> tuple[str, str] | None:
    """Если строка — заголовок раздела, вернуть (раздел, текст после заголовка).

    Поддерживаются формы: «## Опыт работы», «EXPERIENCE», «Навыки: Python, SQL»,
    hh.ru-стиль «Опыт работы — 6 лет 2 месяца».
    """
    raw = line.strip()
    if not raw:
        return None
    body = re.sub(r"^#{1,6}\s*", "", raw).strip().strip("*_").strip()
    # «Навыки: Python, SQL» — заголовок с содержимым в той же строке
    m = re.match(r"^([^:]{2,40}):\s*(.*)$", body)
    if m:
        key = re.sub(r"\s+", " ", m.group(1).strip("*_ ")).lower()
        if key in _HEADING_LOOKUP:
            return _HEADING_LOOKUP[key], m.group(2).strip().strip("*_").strip()
    if len(raw) > 60:
        return None
    key = _normalize_heading(raw)
    if key in _HEADING_LOOKUP:
        return _HEADING_LOOKUP[key], ""
    for phrase, section in _HEADING_LOOKUP.items():
        if key.startswith(phrase):
            rest = key[len(phrase) :].strip()
            if rest and rest[0] in "—–-(" and re.search(r"\d", rest):
                return section, rest.lstrip("—–-( ").rstrip(")")
    return None


def detect_heading(line: str) -> str | None:
    """Вернуть имя раздела, если строка — заголовок раздела резюме."""
    parsed = parse_heading(line)
    return parsed[0] if parsed else None


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w)")
_LINK = re.compile(
    r"(?:https?://\S+|(?:www\.)?(?:linkedin\.com|github\.com|gitlab\.com|t\.me|hh\.ru)/\S+"
    r"|(?<![\w.])@[A-Za-z][\w]{3,})"
)


def extract_contacts(text: str) -> Contacts:
    emails = list(dict.fromkeys(_EMAIL.findall(text)))
    phones: list[str] = []
    for m in _PHONE.finditer(text):
        cand = m.group(0).strip()
        digits = re.sub(r"\D", "", cand)
        if 10 <= len(digits) <= 15 and not re.fullmatch(r"(19|20)\d{2}\D+(19|20)\d{2}", cand):
            phones.append(cand)
    email_tails = {"@" + e.split("@", 1)[1].split(".")[0] for e in emails}
    links = [
        lk.rstrip(".,;)")
        for lk in _LINK.findall(text)
        if lk not in email_tails and not any(lk in e for e in emails)
    ]
    return Contacts(
        emails=emails, phones=list(dict.fromkeys(phones)), links=list(dict.fromkeys(links))
    )


def _guess_name(header: str) -> str | None:
    for line in header.splitlines():
        s = re.sub(r"^#{1,6}\s*", "", line).strip().strip("*")
        if not s:
            continue
        words = s.split()
        if (
            2 <= len(words) <= 4
            and not re.search(r"[\d@:/|]", s)
            and all(w[:1].isupper() for w in words)
        ):
            return s
        return None
    return None


def detect_language(text: str) -> str:
    cyr = len(re.findall(r"[А-Яа-яЁё]", text))
    lat = len(re.findall(r"[A-Za-z]", text))
    return "ru" if cyr >= 0.3 * max(1, cyr + lat) else "en"


def split_sections(text: str) -> dict[str, str]:
    """Разбить резюме на разделы по заголовкам.

    Текст до первого заголовка (ФИО, должность, контакты) попадает в ``contacts``.
    Если заголовки не найдены вовсе, весь текст кладётся в ``other``.
    """
    buckets: dict[str, list[str]] = {}
    current = "contacts"
    found_heading = False
    for line in text.splitlines():
        parsed = parse_heading(line)
        if parsed is not None:
            current, inline = parsed
            found_heading = True
            buckets.setdefault(current, [])
            if inline:
                buckets[current].append(inline)
            continue
        buckets.setdefault(current, []).append(line)
    if not found_heading:
        return {"other": text.strip()} if text.strip() else {}
    return {k: "\n".join(v).strip() for k, v in buckets.items() if "\n".join(v).strip()}


def parse_resume(text: str, doc: DocumentInfo | None = None) -> Resume:
    sections = split_sections(text)
    if "contacts" not in sections:
        # Контакты могли быть внизу документа без заголовка — ищем по всему тексту.
        found = extract_contacts(text)
        if found.emails or found.phones:
            sections = {
                "contacts": "\n".join(found.emails + found.phones + found.links),
                **sections,
            }
    contacts = extract_contacts(sections.get("contacts", "") or text)
    if not (contacts.emails or contacts.phones):
        contacts = extract_contacts(text)
    contacts.name = _guess_name(sections.get("contacts", text))
    return Resume(
        raw_text=text,
        sections=sections,
        contacts=contacts,
        language=detect_language(text),  # type: ignore[arg-type]
        doc=doc,
    )


def load_resume(path: str | Path) -> Resume:
    doc = load_document(path)
    return parse_resume(doc.text, doc)


def load_resume_bytes(data: bytes, filename: str) -> Resume:
    doc = load_document_bytes(data, filename)
    return parse_resume(doc.text, doc)
