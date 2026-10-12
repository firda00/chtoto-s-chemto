"""Анализ партии: оценки до/после хода, потери в сантипешках, классификация, точность.

Формулы
-------
* **cp loss** = ``max(0, eval_before - eval_after)``, обе оценки — с точки зрения ходившего,
  обрезаны до ±1000 (мат = ±1000). ``eval_before`` — оценка позиции до хода (т.е. при лучшей
  игре), ``eval_after`` — оценка позиции после сыгранного хода.
* **Win%** (lichess): ``50 + 50 * (2 / (1 + exp(-0.00368208 * cp)) - 1)``.
* **Точность хода** (lichess): ``103.1668 * exp(-0.04354 * (win_before - win_after)) - 3.1669 + 1``,
  обрезано до [0, 100]; если win% не упал — 100.
* **Точность за партию** (lichess): среднее из взвешенного по волатильности среднего и
  гармонического среднего точностей ходов игрока.
* **ACPL** — средний cp loss игрока.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from enum import StrEnum

import chess
import chess.pgn
from pydantic import BaseModel, Field

from .engine import Engine, EngineEval
from .motifs import Motif, detect_motifs
from .pgn import game_headers, game_title, iter_positions

EVAL_CAP = 1000


class MoveClass(StrEnum):
    BEST = "best"
    EXCELLENT = "excellent"
    GOOD = "good"
    INACCURACY = "inaccuracy"
    MISTAKE = "mistake"
    BLUNDER = "blunder"


CLASS_ORDER = list(MoveClass)  # от лучшего к худшему

# Пороговые значения cp loss (строго больше → класс)
THRESHOLDS = {
    MoveClass.EXCELLENT: 0,  # 0 < loss ≤ 20
    MoveClass.GOOD: 20,  # 20 < loss ≤ 50
    MoveClass.INACCURACY: 50,  # 50 < loss ≤ 100
    MoveClass.MISTAKE: 100,  # 100 < loss ≤ 300
    MoveClass.BLUNDER: 300,  # loss > 300
}


# ---------------------------------------------------------------------------- формулы


def centipawn_loss(eval_before: int, eval_after: int, cap: int = EVAL_CAP) -> int:
    """Потеря в сантипешках с точки зрения ходившего (оценки обрезаются до ±cap)."""
    b = max(-cap, min(cap, eval_before))
    a = max(-cap, min(cap, eval_after))
    return max(0, b - a)


def win_percent(cp: float) -> float:
    """Вероятность выигрыша (0..100) по оценке в сантипешках (формула lichess)."""
    cp = max(-EVAL_CAP, min(EVAL_CAP, cp))
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


def move_accuracy(win_before: float, win_after: float) -> float:
    """Точность хода по падению win% (формула lichess, с бонусом +1 за неопределённость)."""
    if win_after >= win_before:
        return 100.0
    diff = win_before - win_after
    raw = 103.1668100711649 * math.exp(-0.04354415386753951 * diff) - 3.166924740191411
    return max(0.0, min(100.0, raw + 1))


def _harmonic_mean(xs: list[float]) -> float:
    return len(xs) / sum(1 / max(1.0, x) for x in xs)


def game_accuracy(white_win_percents: list[float]) -> tuple[float | None, float | None]:
    """Точность белых и чёрных за партию (lichess ``AccuracyPercent.gameAccuracy``).

    ``white_win_percents`` — win% белых для *всех* позиций партии, включая начальную.
    """
    n = len(white_win_percents)
    if n < 2:
        return None, None
    window = max(2, min(8, n // 10))
    windows: list[list[float]] = [white_win_percents[:window]] * min(window - 2, n)
    windows += [white_win_percents[i : i + window] for i in range(0, max(1, n - window + 1))]
    weights = [max(0.5, min(12.0, statistics.pstdev(w))) for w in windows]
    per_color: dict[bool, list[tuple[float, float]]] = {chess.WHITE: [], chess.BLACK: []}
    for i in range(n - 1):
        prev, nxt = white_win_percents[i], white_win_percents[i + 1]
        color = chess.WHITE if i % 2 == 0 else chess.BLACK
        if color == chess.WHITE:
            acc = move_accuracy(prev, nxt)
        else:
            acc = move_accuracy(100 - prev, 100 - nxt)
        weight = weights[i] if i < len(weights) else weights[-1]
        per_color[color].append((acc, weight))

    def color_accuracy(items: list[tuple[float, float]]) -> float | None:
        if not items:
            return None
        total_w = sum(w for _, w in items)
        weighted = sum(a * w for a, w in items) / total_w
        harmonic = _harmonic_mean([a for a, _ in items])
        return (weighted + harmonic) / 2

    return color_accuracy(per_color[chess.WHITE]), color_accuracy(per_color[chess.BLACK])


def classify_move(
    cp_loss: int,
    *,
    is_best: bool = False,
    mate_before: int | None = None,
    mate_after: int | None = None,
    eval_after: int | None = None,
) -> MoveClass:
    """Классифицировать ход.

    ``mate_before`` / ``mate_after`` — мат в N с точки зрения ходившего (>0 — он матует,
    <0 — матуют его, ``0`` в ``mate_after`` — мат поставлен этим ходом).
    ``eval_after`` — оценка после хода (сантипешки, точка зрения ходившего).

    Учёт матов:
      1. сыгран лучший ход движка → ``best``;
      2. ход допустил форсированный мат (а до хода мата против нас не было) → ``blunder``;
      3. ход упустил форсированный мат: при оценке после хода < +3 → ``blunder``,
         < +7 → ``mistake``, иначе минимум ``inaccuracy``;
      4. иначе — по порогам cp loss: 0 → best, ≤20 excellent, ≤50 good, ≤100 inaccuracy,
         ≤300 mistake, >300 blunder.
    """
    if is_best:
        return MoveClass.BEST
    if mate_after is not None and mate_after < 0 and (mate_before is None or mate_before >= 0):
        return MoveClass.BLUNDER
    had_mate = mate_before is not None and mate_before > 0
    keeps_mate = mate_after is not None and mate_after >= 0
    if had_mate and not keeps_mate:
        ev = eval_after if eval_after is not None else 0
        if ev < 300:
            return MoveClass.BLUNDER
        if ev < 700:
            return MoveClass.MISTAKE
        by_loss = _by_threshold(cp_loss)
        return max(by_loss, MoveClass.INACCURACY, key=CLASS_ORDER.index)
    return _by_threshold(cp_loss)


def _by_threshold(cp_loss: int) -> MoveClass:
    if cp_loss <= 0:
        return MoveClass.BEST
    result = MoveClass.EXCELLENT
    for cls, threshold in THRESHOLDS.items():
        if cp_loss > threshold:
            result = cls
    return result


# ---------------------------------------------------------------------------- модели


class MoveAnalysis(BaseModel):
    ply: int
    move_number: int
    color: str  # "white" | "black"
    san: str
    uci: str
    fen_before: str
    fen_after: str
    eval_before: int  # cp, точка зрения ходившего, обрезано до ±1000
    eval_after: int
    mate_before: int | None = None  # мат в N, точка зрения ходившего
    mate_after: int | None = None  # 0 — мат поставлен этим ходом
    cp_loss: int
    win_before: float
    win_after: float
    accuracy: float
    classification: MoveClass
    best_move_san: str | None = None
    best_move_uci: str | None = None
    best_line_san: list[str] = Field(default_factory=list)
    motifs: list[Motif] = Field(default_factory=list)
    explanation: str | None = None

    @property
    def label(self) -> str:
        dots = "." if self.color == "white" else "..."
        return f"{self.move_number}{dots} {self.san}"

    @property
    def eval_swing(self) -> int:
        return self.eval_after - self.eval_before


class PlayerStats(BaseModel):
    name: str
    color: str
    accuracy: float | None = None
    acpl: float | None = None
    moves: int = 0
    counts: dict[str, int] = Field(default_factory=dict)


class GameAnalysis(BaseModel):
    title: str
    headers: dict[str, str]
    engine: str
    depth: int | None = None
    result: str = "*"
    white: PlayerStats
    black: PlayerStats
    moves: list[MoveAnalysis]
    white_evals: list[int] = Field(default_factory=list)  # оценка белых для каждой позиции

    def critical_moves(self, min_class: MoveClass = MoveClass.MISTAKE) -> list[MoveAnalysis]:
        idx = CLASS_ORDER.index(min_class)
        return [m for m in self.moves if CLASS_ORDER.index(m.classification) >= idx]


# ---------------------------------------------------------------------------- анализ


def _pv_to_san(board: chess.Board, pv: list[str], limit: int = 8) -> list[str]:
    b = board.copy(stack=False)
    out: list[str] = []
    for uci in pv[:limit]:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in b.legal_moves:
            break
        out.append(b.san(move))
        b.push(move)
    return out


def _mover_mate(ev: EngineEval, board_after_or_none: chess.Board | None, color: bool) -> int | None:
    if board_after_or_none is not None and board_after_or_none.is_checkmate():
        return 0 if board_after_or_none.turn != color else None
    return ev.pov_mate(color)


def analyse_game(
    game: chess.pgn.Game,
    engine: Engine,
    depth: int | None = None,
    time: float | None = None,
    progress: Callable[[int, int], None] | None = None,
    detect: bool = True,
) -> GameAnalysis:
    """Полный анализ партии: одна оценка движком на каждую позицию (N ходов → N+1 вызов)."""
    steps = list(iter_positions(game))
    boards = [s.board_before for s in steps]
    final = steps[-1].board_after if steps else game.board()
    boards.append(final)

    evals: list[EngineEval] = []
    for i, board in enumerate(boards):
        evals.append(engine.analyse(board, depth=depth, time=time))
        if progress:
            progress(i + 1, len(boards))

    moves: list[MoveAnalysis] = []
    white_evals = [ev.pov_cp(chess.WHITE, EVAL_CAP) for ev in evals]
    for i, step in enumerate(steps):
        before, after = evals[i], evals[i + 1]
        color = step.color
        board_after = boards[i + 1]
        eb = before.pov_cp(color, EVAL_CAP)
        ea = after.pov_cp(color, EVAL_CAP)
        loss = centipawn_loss(eb, ea)
        mate_b = before.pov_mate(color)
        mate_a = _mover_mate(after, board_after, color)
        is_best = before.best_move == step.move.uci()
        cls = classify_move(
            loss, is_best=is_best, mate_before=mate_b, mate_after=mate_a, eval_after=ea
        )
        wb, wa = win_percent(eb), win_percent(ea)
        motifs: list[Motif] = []
        if detect and cls in (MoveClass.INACCURACY, MoveClass.MISTAKE, MoveClass.BLUNDER):
            motifs = detect_motifs(step.board_before, step.move, before, after)
        best_line = _pv_to_san(step.board_before, before.pv)
        moves.append(
            MoveAnalysis(
                ply=step.ply,
                move_number=step.move_number,
                color="white" if color == chess.WHITE else "black",
                san=step.san,
                uci=step.move.uci(),
                fen_before=step.board_before.fen(),
                fen_after=board_after.fen(),
                eval_before=eb,
                eval_after=ea,
                mate_before=mate_b,
                mate_after=mate_a,
                cp_loss=loss,
                win_before=round(wb, 2),
                win_after=round(wa, 2),
                accuracy=round(move_accuracy(wb, wa), 2),
                classification=cls,
                best_move_san=best_line[0] if best_line else None,
                best_move_uci=before.best_move,
                best_line_san=best_line,
                motifs=motifs,
            )
        )

    headers = game_headers(game)
    first_color = steps[0].color if steps else chess.WHITE
    # Для точности за партию ряд win% должен начинаться с хода белых.
    wp = [win_percent(cp) for cp in white_evals]
    if first_color == chess.BLACK:
        wp = [wp[0], *wp]
    white_acc, black_acc = game_accuracy(wp)

    def stats(color_name: str, name: str, acc: float | None) -> PlayerStats:
        mine = [m for m in moves if m.color == color_name]
        counts = {c.value: 0 for c in MoveClass}
        for m in mine:
            counts[m.classification.value] += 1
        return PlayerStats(
            name=name,
            color=color_name,
            accuracy=round(acc, 1) if acc is not None else None,
            acpl=round(sum(m.cp_loss for m in mine) / len(mine), 1) if mine else None,
            moves=len(mine),
            counts=counts,
        )

    return GameAnalysis(
        title=game_title(game),
        headers=headers,
        engine=engine.name,
        depth=depth,
        result=headers.get("Result", "*"),
        white=stats("white", headers.get("White", "White"), white_acc),
        black=stats("black", headers.get("Black", "Black"), black_acc),
        moves=moves,
        white_evals=white_evals,
    )
