import itertools

import pytest

from rag_system.chunking import (
    FixedSizeChunker,
    MarkdownChunker,
    load_documents,
    parse_frontmatter,
    split_fixed,
)
from rag_system.schemas import Document

from .conftest import DOCS_DIR


def test_split_fixed_respects_size_and_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(300))
    pieces = split_fixed(text, chunk_size=100, overlap=20)
    assert len(pieces) > 5
    assert all(len(p) <= 100 for p in pieces)
    # соседние куски перекрываются: последнее слово куска встречается в начале следующего
    for a, b in itertools.pairwise(pieces):
        assert a.split()[-1] in b.split()[:6]
    # ни одно слово не разрезано
    assert all(w.startswith("word") for p in pieces for w in p.split())


def test_split_fixed_short_text_and_validation() -> None:
    assert split_fixed("hello world", 100, 10) == ["hello world"]
    assert split_fixed("   ", 100, 10) == []
    with pytest.raises(ValueError):
        split_fixed("abc", 10, 10)


def test_fixed_chunker_keeps_metadata() -> None:
    doc = Document(id="d", text="a " * 400, metadata={"collection": "x", "source": "d.md"})
    chunks = FixedSizeChunker(chunk_size=200, overlap=40).split(doc)
    assert len(chunks) >= 4
    assert {c.collection for c in chunks} == {"x"}
    assert len({c.id for c in chunks}) == len(chunks)
    assert chunks[0].metadata["chunker"] == "fixed"


def test_frontmatter() -> None:
    meta, body = parse_frontmatter("---\ntitle: T\ncollection: pricing\n---\n# H\ntext")
    assert meta == {"title": "T", "collection": "pricing"}
    assert body.startswith("# H")


def test_markdown_chunker_heading_aware() -> None:
    text = "# Root\nintro\n\n## A\nalpha text\n\n### A1\ndeep text\n\n## B\nbeta text\n"
    chunks = MarkdownChunker(max_chars=500).split(Document(id="d", text=text))
    assert [c.heading for c in chunks] == ["Root", "Root > A", "Root > A > A1", "Root > B"]
    assert chunks[2].text.startswith("Root > A > A1\n")
    assert "deep text" in chunks[2].text
    assert "beta" not in chunks[2].text


def test_markdown_chunker_splits_long_sections_and_ignores_code_headings() -> None:
    body = "## Big\n" + "lorem ipsum " * 200 + "\n```\n# not a heading\n```\n"
    chunks = MarkdownChunker(max_chars=300, overlap=30).split(Document(id="d", text=body))
    assert len(chunks) > 3
    assert all(c.heading == "Big" for c in chunks)
    assert all(len(c.text) <= 300 for c in chunks)


def test_load_documents_from_data_dir() -> None:
    docs = load_documents(DOCS_DIR)
    assert 5 <= len(docs) <= 8
    collections = {d.metadata["collection"] for d in docs}
    assert {"pricing", "schedule", "faq", "policies"} <= collections
    assert all(d.metadata["source"].endswith(".md") for d in docs)
