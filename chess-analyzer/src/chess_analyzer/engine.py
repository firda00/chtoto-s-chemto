"""Движки оценки позиций.

Все движки реализуют протокол :class:`Engine` и возвращают :class:`EngineEval`.
Оценка (``score_cp`` / ``mate``) всегда хранится **с точки зрения белых**:
``score_cp=+150`` — белые лучше на полторы пешки, ``mate=+3`` — белые матуют в 3 хода,
``mate=-2`` — чёрные матуют в 2 хода.

* :class:`StockfishEngine` — локальный бинарник через ``chess.engine.SimpleEngine`` (UCI).
* :class:`ApiEngine` — удалённый «Stockfish API» по HTTP (формат stockfish.online v2).
* :class:`FakeEngine` — детерминированный оффлайн-движок для тестов и демо.
"""

from __future__ import annotations

import os
import shutil
from typing import Protocol, runtime_checkable

import chess
import chess.engine
import httpx
from pydantic import BaseModel, Field

MATE_CP = 100_000  # внутреннее «бесконечное» значение для мата
DEFAULT_API_URL = "https://stockfish.online/api/s/v2.php"


class EngineEval(BaseModel):
    """Результат анализа позиции (с точки зрения белых)."""

    score_cp: int | None = None
    mate: int | None = None
    best_move: str | None = None  # UCI
    pv: list[str] = Field(default_factory=list)  # UCI
    depth: int | None = None

    def pov_cp(self, color: chess.Color, cap: int = 1000) -> int:
        """Оценка в сантипешках с точки зрения ``color``; мат → ±cap; всё обрезано до ±cap."""
        if self.mate is not None:
            value = cap if self.mate > 0 else -cap
        else:
            value = max(-cap, min(cap, self.score_cp or 0))
        return value if color == chess.WHITE else -value

    def pov_mate(self, color: chess.Color) -> int | None:
        """Мат в N с точки зрения ``color`` (>0 — ``color`` матует, <0 — матуют его)."""
        if self.mate is None:
            return None
        return self.mate if color == chess.WHITE else -self.mate


@runtime_checkable
class Engine(Protocol):
    name: str

    def analyse(
        self, board: chess.Board, depth: int | None = None, time: float | None = None
    ) -> EngineEval: ...

    def close(self) -> None: ...


def terminal_eval(board: chess.Board) -> EngineEval | None:
    """Оценка для завершённой позиции (мат/пат/недостаточно материала) или ``None``."""
    if board.is_checkmate():
        # Сторона, которая должна ходить, получила мат.
        return EngineEval(score_cp=-MATE_CP if board.turn == chess.WHITE else MATE_CP, depth=0)
    if board.is_stalemate() or board.is_insufficient_material():
        return EngineEval(score_cp=0, depth=0)
    return None


# --------------------------------------------------------------------------- Stockfish (UCI)


def find_stockfish(path: str | None = None) -> str | None:
    """Путь к Stockfish: аргумент → ``STOCKFISH_PATH`` → поиск в PATH."""
    for candidate in (path, os.environ.get("STOCKFISH_PATH")):
        if candidate and os.path.isfile(candidate):
            return candidate
    for name in ("stockfish", "stockfish.exe", "stockfish-windows-x86-64-avx2.exe"):
        found = shutil.which(name)
        if found:
            return found
    return None


class StockfishEngine:
    """Локальный Stockfish через UCI (``chess.engine.SimpleEngine``)."""

    name = "stockfish"

    def __init__(self, path: str | None = None, threads: int = 1, hash_mb: int = 64):
        resolved = find_stockfish(path)
        if not resolved:
            raise FileNotFoundError(
                "Stockfish не найден: укажите STOCKFISH_PATH или добавьте stockfish в PATH"
            )
        self.path = resolved
        self._engine = chess.engine.SimpleEngine.popen_uci(resolved)
        self._engine.configure({"Threads": threads, "Hash": hash_mb})

    def analyse(
        self, board: chess.Board, depth: int | None = None, time: float | None = None
    ) -> EngineEval:
        term = terminal_eval(board)
        if term is not None:
            return term
        if time is not None:
            limit = chess.engine.Limit(time=time)
        else:
            limit = chess.engine.Limit(depth=depth or 14)
        info = self._engine.analyse(board, limit)
        score = info["score"].white()
        pv = [m.uci() for m in info.get("pv", [])]
        return EngineEval(
            score_cp=score.score(),
            mate=score.mate(),
            best_move=pv[0] if pv else None,
            pv=pv,
            depth=info.get("depth"),
        )

    def close(self) -> None:
        self._engine.quit()


# --------------------------------------------------------------------------- Stockfish HTTP API


class ApiEngine:
    """Удалённый Stockfish по HTTP в стиле https://stockfish.online (API v2).

    Запрос: ``GET {url}?fen=<FEN>&depth=<N>``. Ожидаемый ответ::

        {"success": true, "evaluation": 1.36, "mate": null,
         "bestmove": "bestmove b7b6 ponder f3e5", "continuation": "b7b6 f3e5 h7h6"}

    ``evaluation`` — в пешках с точки зрения белых, ``mate`` — со знаком (+ — матуют белые).
    """

    name = "api"
    MAX_DEPTH = 15  # ограничение публичного stockfish.online

    def __init__(
        self, url: str | None = None, client: httpx.Client | None = None, timeout: float = 30.0
    ):
        self.url = url or os.environ.get("STOCKFISH_API_URL") or DEFAULT_API_URL
        self._client = client or httpx.Client(timeout=timeout)

    def analyse(
        self, board: chess.Board, depth: int | None = None, time: float | None = None
    ) -> EngineEval:
        term = terminal_eval(board)
        if term is not None:
            return term
        d = min(depth or 12, self.MAX_DEPTH)
        resp = self._client.get(self.url, params={"fen": board.fen(), "depth": d})
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success", True):
            raise RuntimeError(f"Stockfish API error: {data.get('data') or data}")
        mate = data.get("mate")
        evaluation = data.get("evaluation")
        pv = (data.get("continuation") or "").split()
        parts = (data.get("bestmove") or "").split()
        best = None
        if len(parts) >= 2 and parts[0] == "bestmove":
            best = parts[1]
        elif parts:
            best = parts[0]
        if best and (not pv or pv[0] != best):
            pv = [best]
        return EngineEval(
            score_cp=None if mate is not None or evaluation is None else round(evaluation * 100),
            mate=int(mate) if mate is not None else None,
            best_move=best,
            pv=pv,
            depth=d,
        )

    def close(self) -> None:
        self._client.close()


# --------------------------------------------------------------------------- Fake engine

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Маленькая таблица «центральности» (индекс = номер поля, a1=0); симметрична по вертикали.
_CENTER = [
    0, 1, 2, 3, 3, 2, 1, 0,
    1, 2, 3, 4, 4, 3, 2, 1,
    2, 3, 4, 5, 5, 4, 3, 2,
    3, 4, 5, 6, 6, 5, 4, 3,
    3, 4, 5, 6, 6, 5, 4, 3,
    2, 3, 4, 5, 5, 4, 3, 2,
    1, 2, 3, 4, 4, 3, 2, 1,
    0, 1, 2, 3, 3, 2, 1, 0,
]  # fmt: skip


def static_eval(board: chess.Board) -> int:
    """Материал + небольшой бонус за центр/продвижение пешек (с точки зрения белых)."""
    score = 0
    for sq, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]
        rank = chess.square_rank(sq) if piece.color == chess.WHITE else 7 - chess.square_rank(sq)
        if piece.piece_type == chess.PAWN:
            value += 4 * rank
        elif piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            value += 3 * _CENTER[sq]
        elif piece.piece_type == chess.KING:
            value -= 2 * _CENTER[sq]  # король в центре — плохо
        score += value if piece.color == chess.WHITE else -value
    return score


def _move_order_key(board: chess.Board, move: chess.Move) -> tuple[int, str]:
    """Сначала взятия (MVV-LVA), потом превращения, потом остальное по UCI — детерминированно."""
    if board.is_capture(move):
        victim = board.piece_type_at(move.to_square) or chess.PAWN  # en passant
        attacker = board.piece_type_at(move.from_square) or chess.PAWN
        return (-10_000 - PIECE_VALUES[victim] * 10 + PIECE_VALUES[attacker] // 10, move.uci())
    if move.promotion:
        return (-5_000, move.uci())
    return (0, move.uci())


class FakeEngine:
    """Детерминированный оффлайн-движок: негамакс с альфа-бета на глубину ``max_depth``.

    Оценка листа — :func:`static_eval` (материал + центр). Ходы перебираются в фиксированном
    порядке (сначала взятия, затем по UCI), поэтому при равных оценках выбирается взятие
    или первый легальный ход. Результат полностью воспроизводим. Мат обнаруживается,
    если он укладывается в глубину перебора (depth=2 видит мат в 1, depth=4 — мат в 2).
    Нечётная глубина округляется вниз до чётной (иначе соседние позиции оцениваются
    с разным «горизонтом» и cp loss систематически завышается).
    """

    name = "fake"

    def __init__(self, max_depth: int = 2):
        self.max_depth = max_depth

    def _negamax(
        self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int
    ) -> tuple[int, list[chess.Move]]:
        if depth == 0:
            if board.is_check() and board.is_checkmate():
                return -(MATE_CP - ply), []
            sign = 1 if board.turn == chess.WHITE else -1
            return sign * static_eval(board), []
        moves = list(board.legal_moves)
        if not moves:
            return (-(MATE_CP - ply), []) if board.is_check() else (0, [])
        moves.sort(key=lambda m: _move_order_key(board, m))
        best_score = -MATE_CP - 1
        best_line: list[chess.Move] = []
        for move in moves:
            board.push(move)
            score, line = self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            score = -score
            if score > best_score:
                best_score, best_line = score, [move, *line]
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best_score, best_line

    def analyse(
        self, board: chess.Board, depth: int | None = None, time: float | None = None
    ) -> EngineEval:
        term = terminal_eval(board)
        if term is not None:
            return term
        d = max(1, min(depth or self.max_depth, self.max_depth))
        if d > 1 and d % 2:
            d -= 1  # чётная глубина: у обеих сторон одинаковый «горизонт», иначе оценки скачут
        score, line = self._negamax(board.copy(stack=False), d, -MATE_CP - 1, MATE_CP + 1, 0)
        sign = 1 if board.turn == chess.WHITE else -1
        mate: int | None = None
        cp: int | None = sign * score
        if abs(score) >= MATE_CP - 100:
            plies = MATE_CP - abs(score)
            moves_to_mate = (plies + 1) // 2
            mate = sign * (moves_to_mate if score > 0 else -moves_to_mate)
            cp = None
        pv = [m.uci() for m in line]
        return EngineEval(score_cp=cp, mate=mate, best_move=pv[0] if pv else None, pv=pv, depth=d)

    def close(self) -> None:
        pass


def make_engine(kind: str = "auto", fake_depth: int = 2) -> Engine:
    """Фабрика: ``fake`` | ``stockfish`` | ``api`` | ``auto`` (Stockfish или fake)."""
    kind = kind.lower()
    if kind == "auto":
        kind = "stockfish" if find_stockfish() else "fake"
    if kind == "fake":
        return FakeEngine(max_depth=fake_depth)
    if kind == "stockfish":
        return StockfishEngine()
    if kind == "api":
        return ApiEngine()
    raise ValueError(f"Неизвестный движок: {kind!r} (ожидается fake|stockfish|api|auto)")
