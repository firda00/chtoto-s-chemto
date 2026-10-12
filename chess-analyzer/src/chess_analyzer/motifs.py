"""Простые эвристики тактических мотивов (на основе карт атак python-chess).

Это **не** полноценный тактический анализатор, а набор дешёвых и объяснимых правил:

* ``hanging_piece`` — после хода фигура ходившего атакована и либо не защищена, либо
  атакована более дешёвой фигурой (связки атакующих и промежуточные ходы не учитываются).
* ``missed_mate`` — у ходившего был форсированный мат (по движку или найденный перебором
  мат в 1), а сыгранный ход его упускает.
* ``allowed_mate`` — после хода у соперника появился форсированный мат (по движку или мат в 1).
* ``missed_fork`` / ``allowed_fork`` — лучший ход (наш / ответ соперника) — «вилка»: ходившая
  фигура в безопасности и атакует ≥2 целей (король, более ценная или незащищённая фигура ≥ коня).
* ``missed_pin`` / ``allowed_pin`` — лучший ход создаёт новую *абсолютную* связку (к королю)
  ходившей фигурой (``Board.is_pinned``; относительные связки к ферзю не ищем).
* ``missed_capture`` — лучший ход берёт фигуру ценой ≥ коня, а сыгран другой ход.
"""

from __future__ import annotations

from typing import Literal

import chess
from pydantic import BaseModel, Field

from .engine import PIECE_VALUES, EngineEval

MotifKind = Literal[
    "hanging_piece",
    "missed_mate",
    "allowed_mate",
    "missed_fork",
    "allowed_fork",
    "missed_pin",
    "allowed_pin",
    "missed_capture",
]


class Motif(BaseModel):
    kind: MotifKind
    squares: list[str] = Field(default_factory=list)
    piece: str | None = None  # имя фигуры (англ.), напр. "queen"
    mate_in: int | None = None
    move_san: str | None = None  # ход, который реализует/реализовал бы мотив


def piece_name(piece_type: chess.PieceType) -> str:
    return chess.piece_name(piece_type)


def value_of(board: chess.Board, square: chess.Square) -> int:
    pt = board.piece_type_at(square)
    return PIECE_VALUES[pt] if pt else 0


def hanging_pieces(
    board: chess.Board, color: chess.Color, min_value: int = 100
) -> list[chess.Square]:
    """Фигуры ``color``, которые можно выгодно взять (без короля), по убыванию ценности."""
    result = []
    for sq in chess.SquareSet(board.occupied_co[color]):
        pt = board.piece_type_at(sq)
        if pt is None or pt == chess.KING or PIECE_VALUES[pt] < min_value:
            continue
        attackers = board.attackers(not color, sq)
        if not attackers:
            continue
        defenders = board.attackers(color, sq)
        cheapest = min(PIECE_VALUES[board.piece_type_at(a)] or 10_000 for a in attackers)
        # король атакует «бесплатно», но не может брать защищённую фигуру
        if not defenders or (0 < cheapest < PIECE_VALUES[pt]):
            result.append(sq)
    result.sort(key=lambda s: -value_of(board, s))
    return result


def find_mate_in_1(board: chess.Board) -> chess.Move | None:
    """Первый (в порядке UCI) ход, дающий мат в 1, или ``None``."""
    for move in sorted(board.legal_moves, key=lambda m: m.uci()):
        board.push(move)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return move
    return None


def fork_targets(board: chess.Board, move: chess.Move) -> list[chess.Square]:
    """Цели «вилки» после ``move`` (≥2) или пустой список."""
    if move not in board.legal_moves:
        return []
    mover = board.turn
    after = board.copy(stack=False)
    after.push(move)
    sq = move.to_square
    moved_value = value_of(after, sq) or 1
    attackers = after.attackers(not mover, sq)
    defenders = after.attackers(mover, sq)
    if attackers:
        cheapest = min(value_of(after, a) or 10_000 for a in attackers)
        if not defenders or cheapest < moved_value:
            return []  # фигуру просто заберут
    targets = []
    for target in after.attacks(sq):
        piece = after.piece_at(target)
        if piece is None or piece.color == mover:
            continue
        val = PIECE_VALUES[piece.piece_type]
        undefended = not after.attackers(not mover, target)
        if piece.piece_type == chess.KING or val > moved_value or (undefended and val >= 300):
            targets.append(target)
    return targets if len(targets) >= 2 else []


def new_pins(board: chess.Board, move: chess.Move) -> list[chess.Square]:
    """Фигуры соперника, которые ``move`` связывает (абсолютно, к королю) ходившей фигурой."""
    if move not in board.legal_moves:
        return []
    opp = not board.turn
    before_pinned = {
        sq for sq in chess.SquareSet(board.occupied_co[opp]) if board.is_pinned(opp, sq)
    }
    after = board.copy(stack=False)
    after.push(move)
    pinned = []
    for sq in chess.SquareSet(after.occupied_co[opp]):
        if after.piece_type_at(sq) == chess.KING or sq in before_pinned:
            continue
        if after.is_pinned(opp, sq) and move.to_square in after.pin(opp, sq):
            pinned.append(sq)
    return pinned


def _san(board: chess.Board, uci: str | None) -> str | None:
    if not uci:
        return None
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        return None
    return board.san(move) if move in board.legal_moves else None


def detect_motifs(
    board_before: chess.Board,
    played: chess.Move,
    eval_before: EngineEval | None = None,
    eval_after: EngineEval | None = None,
) -> list[Motif]:
    """Найти мотивы для сыгранного хода. ``eval_*`` — оценки движка (могут быть ``None``)."""
    mover = board_before.turn
    after = board_before.copy(stack=False)
    after.push(played)
    motifs: list[Motif] = []
    delivered_mate = after.is_checkmate()
    best_uci = eval_before.best_move if eval_before else None
    best_move = chess.Move.from_uci(best_uci) if best_uci else None
    reply_uci = eval_after.best_move if eval_after else None
    reply_move = chess.Move.from_uci(reply_uci) if reply_uci else None

    # --- упущенный мат
    if not delivered_mate:
        mate_before = eval_before.pov_mate(mover) if eval_before else None
        mate_after = eval_after.pov_mate(mover) if eval_after else None
        mate1 = find_mate_in_1(board_before)
        if mate_before is not None and mate_before > 0 and not (mate_after and mate_after > 0):
            motifs.append(
                Motif(
                    kind="missed_mate", mate_in=mate_before, move_san=_san(board_before, best_uci)
                )
            )
        elif mate1 is not None:
            motifs.append(Motif(kind="missed_mate", mate_in=1, move_san=board_before.san(mate1)))

        # --- допущенный мат
        opp_mate = eval_after.pov_mate(not mover) if eval_after else None
        opp_mate1 = find_mate_in_1(after)
        if opp_mate is not None and opp_mate > 0:
            motifs.append(
                Motif(kind="allowed_mate", mate_in=opp_mate, move_san=_san(after, reply_uci))
            )
        elif opp_mate1 is not None:
            motifs.append(Motif(kind="allowed_mate", mate_in=1, move_san=after.san(opp_mate1)))

    # --- висящая фигура
    for sq in hanging_pieces(after, mover, min_value=300)[:2]:
        motifs.append(
            Motif(
                kind="hanging_piece",
                squares=[chess.square_name(sq)],
                piece=piece_name(after.piece_type_at(sq)),
            )
        )

    # --- упущенные вилка/связка/взятие (лучший ход != сыгранный)
    if best_move is not None and best_move != played and best_move in board_before.legal_moves:
        best_san = board_before.san(best_move)
        targets = fork_targets(board_before, best_move)
        if targets:
            motifs.append(
                Motif(
                    kind="missed_fork",
                    squares=[chess.square_name(s) for s in targets],
                    piece=piece_name(board_before.piece_type_at(best_move.from_square)),
                    move_san=best_san,
                )
            )
        pins = new_pins(board_before, best_move)
        if pins:
            motifs.append(
                Motif(
                    kind="missed_pin",
                    squares=[chess.square_name(s) for s in pins],
                    piece=piece_name(board_before.piece_type_at(pins[0])),
                    move_san=best_san,
                )
            )
        if board_before.is_capture(best_move) and not any(m.kind == "missed_mate" for m in motifs):
            victim = board_before.piece_type_at(best_move.to_square)
            if victim and PIECE_VALUES[victim] >= 300:
                motifs.append(
                    Motif(
                        kind="missed_capture",
                        squares=[chess.square_name(best_move.to_square)],
                        piece=piece_name(victim),
                        move_san=best_san,
                    )
                )

    # --- допущенные вилка/связка (лучший ответ соперника)
    if reply_move is not None and reply_move in after.legal_moves:
        reply_san = after.san(reply_move)
        targets = fork_targets(after, reply_move)
        if targets:
            motifs.append(
                Motif(
                    kind="allowed_fork",
                    squares=[chess.square_name(s) for s in targets],
                    piece=piece_name(after.piece_type_at(reply_move.from_square)),
                    move_san=reply_san,
                )
            )
        pins = new_pins(after, reply_move)
        if pins:
            motifs.append(
                Motif(
                    kind="allowed_pin",
                    squares=[chess.square_name(s) for s in pins],
                    piece=piece_name(after.piece_type_at(pins[0])),
                    move_san=reply_san,
                )
            )
    return motifs
