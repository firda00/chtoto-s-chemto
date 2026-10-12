"""Выбор парсера по расширению файла или по сигнатуре содержимого."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..models import ParseResult
from .csv_parser import parse_csv
from .pdf_parser import parse_pdf

Parser = Callable[..., ParseResult]

PARSERS_BY_EXT: dict[str, Parser] = {
    ".csv": parse_csv,
    ".txt": parse_csv,
    ".tsv": parse_csv,
    ".pdf": parse_pdf,
}


def pick_parser(path: str | Path) -> Parser:
    path = Path(path)
    parser = PARSERS_BY_EXT.get(path.suffix.lower())
    if parser is not None:
        return parser
    with path.open("rb") as fh:
        head = fh.read(8)
    return parse_pdf if head.startswith(b"%PDF") else parse_csv


def load_statement(path: str | Path, *, default_currency: str = "UZS") -> ParseResult:
    """Главная точка входа: распарсить выписку любого поддерживаемого формата."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pick_parser(path)(path, default_currency=default_currency)
