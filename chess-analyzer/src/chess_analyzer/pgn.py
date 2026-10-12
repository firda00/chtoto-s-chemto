"""Чтение PGN: файл или строка, несколько партий, заголовки, обход позиций."""

from __future__ import annotations

import io
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn


class PGNError(ValueError):
    """PGN не удалось разобрать или в нём нелегальные ходы."""


@dataclass(frozen=True)
class PositionStep:
    """Один полуход партии: позиция *до* хода и сам ход."""

    ply: int  # 1-based номер полухода
    move_number: int  # номер хода в нотации (1. e4 e5 → оба ход 1)
    color: chess.Color  # кто ходил
    board_before: chess.Board
    move: chess.Move
    san: str

    @property
    def board_after(self) -> chess.Board:
        b = self.board_before.copy(stack=False)
        b.push(self.move)
        return b


def _read_text(source: str | Path) -> str:
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8-sig")
    # Строка: если это существующий путь к файлу — читаем файл, иначе считаем PGN-текстом.
    if "\n" not in source and len(source) < 1024:
        p = Path(source)
        try:
            is_file = p.is_file()
        except (OSError, ValueError):
            is_file = False
        if is_file:
            return p.read_text(encoding="utf-8-sig")
        if p.suffix.lower() == ".pgn":
            raise PGNError(f"Файл не найден: {source}")
    return source


def load_games(source: str | Path, strict: bool = True) -> list[chess.pgn.Game]:
    """Загрузить все партии из файла или PGN-строки.

    ``strict=True`` — бросать :class:`PGNError`, если python-chess нашёл ошибки
    (нелегальный ход и т.п.).
    """
    text = _read_text(source)
    stream = io.StringIO(text)
    games: list[chess.pgn.Game] = []
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        if strict and game.errors:
            raise PGNError(f"Ошибка в партии #{len(games) + 1}: {game.errors[0]}")
        is_empty = game.next() is None and "FEN" not in game.headers
        if is_empty and game.headers.get("White", "?") == "?" == game.headers.get("Black", "?"):
            continue  # пустая «партия» (мусор между играми)
        games.append(game)
    if not games:
        raise PGNError("В PGN не найдено ни одной партии")
    return games


def game_headers(game: chess.pgn.Game) -> dict[str, str]:
    """Заголовки партии как обычный словарь."""
    return dict(game.headers.items())


def game_title(game: chess.pgn.Game) -> str:
    h = game.headers
    white, black = h.get("White", "?"), h.get("Black", "?")
    event = h.get("Event", "?")
    date = h.get("Date", "????.??.??")
    return f"{white} — {black} ({event}, {date})"


def iter_positions(game: chess.pgn.Game) -> Iterator[PositionStep]:
    """Обойти основную линию партии: (позиция до хода, ход, SAN)."""
    board = game.board()
    for ply, move in enumerate(game.mainline_moves(), start=1):
        san = board.san(move)
        yield PositionStep(
            ply=ply,
            move_number=board.fullmove_number,
            color=board.turn,
            board_before=board.copy(stack=False),
            move=move,
            san=san,
        )
        board.push(move)


def count_plies(game: chess.pgn.Game) -> int:
    return sum(1 for _ in game.mainline_moves())
