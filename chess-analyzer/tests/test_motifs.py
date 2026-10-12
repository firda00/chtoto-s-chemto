import chess

from chess_analyzer.engine import EngineEval
from chess_analyzer.motifs import (
    detect_motifs,
    find_mate_in_1,
    fork_targets,
    hanging_pieces,
    new_pins,
)

BACK_RANK = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"


def test_hanging_piece_undefended():
    board = chess.Board("r3k3/8/8/R7/8/8/8/4K3 w - - 0 1")  # ладья a5 под боем ладьи a8
    assert hanging_pieces(board, chess.WHITE) == [chess.A5]


def test_hanging_piece_attacked_by_cheaper_piece():
    board = chess.Board("4k3/8/3p4/4N3/8/8/8/3RK3 w - - 0 1")  # конь e5 под пешкой d6
    assert chess.E5 in hanging_pieces(board, chess.WHITE)


def test_defended_piece_attacked_by_equal_is_not_hanging():
    board = chess.Board("4k3/8/2n5/4N3/8/8/8/4RK2 w - - 0 1")  # конь против коня, e5 защищён ладьёй
    assert hanging_pieces(board, chess.WHITE) == []


def test_find_mate_in_1():
    assert find_mate_in_1(chess.Board(BACK_RANK)) == chess.Move.from_uci("a1a8")
    assert find_mate_in_1(chess.Board()) is None


def test_missed_mate_in_1_without_engine():
    board = chess.Board(BACK_RANK)
    motifs = detect_motifs(board, chess.Move.from_uci("g1f2"))
    missed = [m for m in motifs if m.kind == "missed_mate"]
    assert missed and missed[0].mate_in == 1 and missed[0].move_san == "Ra8#"


def test_missed_mate_from_engine_eval():
    board = chess.Board(BACK_RANK)
    before = EngineEval(mate=1, best_move="a1a8", pv=["a1a8"])
    motifs = detect_motifs(board, chess.Move.from_uci("a1a2"), before, EngineEval(score_cp=500))
    assert any(m.kind == "missed_mate" and m.mate_in == 1 for m in motifs)


def test_delivered_mate_has_no_missed_mate():
    board = chess.Board(BACK_RANK)
    assert detect_motifs(board, chess.Move.from_uci("a1a8")) == []


def test_allowed_mate_in_1():
    board = chess.Board("6k1/5ppp/2p5/8/8/8/8/R5K1 b - - 0 1")
    motifs = detect_motifs(board, chess.Move.from_uci("c6c5"))
    allowed = [m for m in motifs if m.kind == "allowed_mate"]
    assert allowed and allowed[0].move_san == "Ra8#"


def test_fork_detection():
    board = chess.Board("r3k3/8/8/3N4/8/8/8/4K3 w - - 0 1")
    targets = fork_targets(board, chess.Move.from_uci("d5c7"))
    assert set(targets) == {chess.A8, chess.E8}
    assert fork_targets(board, chess.Move.from_uci("d5f4")) == []  # обычный ход — не вилка


def test_missed_fork_motif():
    board = chess.Board("r3k3/8/8/3N4/8/8/8/4K3 w - - 0 1")
    before = EngineEval(score_cp=500, best_move="d5c7", pv=["d5c7"])
    motifs = detect_motifs(board, chess.Move.from_uci("e1e2"), before, EngineEval(score_cp=0))
    fork = [m for m in motifs if m.kind == "missed_fork"]
    assert fork and fork[0].move_san == "Nc7+" and fork[0].piece == "knight"


def test_allowed_fork_motif():
    # Чёрные ставят ладью на a8 и разрешают вилку Nc7+ (король e8 + ладья a8).
    board = chess.Board("4k3/8/8/1N6/8/8/r7/4K3 b - - 0 1")
    after_eval = EngineEval(score_cp=300, best_move="b5c7", pv=["b5c7"])
    motifs = detect_motifs(board, chess.Move.from_uci("a2a8"), EngineEval(score_cp=0), after_eval)
    assert any(m.kind == "allowed_fork" and m.move_san == "Nc7+" for m in motifs)


def test_pin_detection():
    board = chess.Board("4k3/8/2n5/8/8/8/8/4KB2 w - - 0 1")
    assert new_pins(board, chess.Move.from_uci("f1b5")) == [chess.C6]
    assert new_pins(board, chess.Move.from_uci("f1d3")) == []
