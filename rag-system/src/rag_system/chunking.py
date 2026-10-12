"""Загрузка документов и разбиение на фрагменты.

Два чанкера:
* FixedSizeChunker — окно фиксированного размера (в символах) с перекрытием,
  граница окна сдвигается к ближайшему пробелу, чтобы не резать слова.
* MarkdownChunker — режет по заголовкам (#..######), сохраняет путь заголовков
  в metadata["heading"] и в начале текста фрагмента; длинные секции дорезает FixedSizeChunker.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from rag_system.schemas import Chunk, Document

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


class Chunker(Protocol):
    def split(self, doc: Document) -> list[Chunk]: ...


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Простой YAML-подобный frontmatter `key: value` между `---`."""
    m = _FRONTMATTER.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip("\"'")
    return meta, text[m.end() :]


def load_documents(path: str | Path) -> list[Document]:
    """Читает файл или рекурсивно папку (*.md, *.txt) в список Document."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Путь не найден: {p}")
    files = (
        [p]
        if p.is_file()
        else sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_SUFFIXES)
    )
    docs: list[Document] = []
    for f in files:
        raw = f.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(raw)
        meta.setdefault("collection", "default")
        meta["source"] = f.name
        meta["path"] = str(f)
        docs.append(Document(id=f.stem, text=body.strip(), metadata=meta))
    return docs


def _chunk_id(doc_id: str, idx: int, text: str) -> str:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).hexdigest()
    return f"{doc_id}:{idx}:{digest}"


def split_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Режет текст на окна ~chunk_size символов с перекрытием overlap, не разрывая слова."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []
    pieces: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            space = text.rfind(" ", start + chunk_size // 2, end)
            if space != -1:
                end = space
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= n:
            break
        next_start = max(end - overlap, start + 1)
        # не начинаем следующий кусок с середины слова
        if overlap and next_start > 0 and text[next_start - 1] != " ":
            sp = text.find(" ", next_start, end)
            if sp != -1:
                next_start = sp + 1
        start = next_start
    return pieces


class FixedSizeChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 80) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, doc: Document) -> list[Chunk]:
        collection = str(doc.metadata.get("collection", "default"))
        chunks = []
        for i, piece in enumerate(split_fixed(doc.text, self.chunk_size, self.overlap)):
            meta = {**doc.metadata, "position": i, "chunker": "fixed"}
            chunks.append(
                Chunk(
                    id=_chunk_id(doc.id, i, piece),
                    doc_id=doc.id,
                    text=piece,
                    collection=collection,
                    metadata=meta,
                )
            )
        return chunks


class MarkdownChunker:
    """Разбиение по заголовкам Markdown с сохранением иерархии заголовков."""

    def __init__(self, max_chars: int = 800, overlap: int = 80, prefix_heading: bool = True):
        self.max_chars = max_chars
        self.overlap = overlap
        self.prefix_heading = prefix_heading

    def sections(self, text: str) -> list[tuple[list[str], str]]:
        """Возвращает [(путь заголовков, тело секции)]."""
        path: list[tuple[int, str]] = []
        out: list[tuple[list[str], str]] = []
        buf: list[str] = []
        in_code = False

        def flush() -> None:
            body = "\n".join(buf).strip()
            if body:
                out.append(([t for _, t in path], body))
            buf.clear()

        for line in text.splitlines():
            if line.strip().startswith("```"):
                in_code = not in_code
            m = None if in_code else _HEADING.match(line)
            if m:
                flush()
                level = len(m.group(1))
                path = [(lvl, t) for lvl, t in path if lvl < level]
                path.append((level, m.group(2).strip()))
            else:
                buf.append(line)
        flush()
        return out

    def split(self, doc: Document) -> list[Chunk]:
        collection = str(doc.metadata.get("collection", "default"))
        chunks: list[Chunk] = []
        idx = 0
        for heading_path, body in self.sections(doc.text):
            heading = " > ".join(heading_path)
            prefix = f"{heading}\n" if self.prefix_heading and heading else ""
            budget = max(self.max_chars - len(prefix), self.max_chars // 2)
            for piece in split_fixed(body, budget, min(self.overlap, budget // 2)):
                text = prefix + piece
                meta = {**doc.metadata, "heading": heading, "position": idx, "chunker": "markdown"}
                chunks.append(
                    Chunk(
                        id=_chunk_id(doc.id, idx, text),
                        doc_id=doc.id,
                        text=text,
                        collection=collection,
                        metadata=meta,
                    )
                )
                idx += 1
        return chunks


def get_chunker(name: str = "markdown", chunk_size: int = 800, overlap: int = 80) -> Chunker:
    if name == "markdown":
        return MarkdownChunker(max_chars=chunk_size, overlap=overlap)
    if name == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    raise ValueError(f"Неизвестный чанкер: {name!r} (markdown | fixed)")


def chunk_documents(docs: Iterable[Document], chunker: Chunker) -> list[Chunk]:
    return [c for d in docs for c in chunker.split(d)]
