import math

import pytest

from chess_analyzer.analysis import (
    MoveClass,
    centipawn_loss,
    classify_move,
    game_accuracy,
    move_accuracy,
    win_percent,
)

# ------------------------------------------------------------------ cp loss


def test_centipawn_loss_basic():
    assert centipawn_loss(120, -30) == 150
    assert centipawn_loss(-30, 120) == 0  # улучшение — не потеря


def test_centipawn_loss_is_capped():
    assert centipawn_loss(5000, -5000) == 2000
    assert centipawn_loss(1000, 800) == 200


# ------------------------------------------------------------------ классификация


@pytest.mark.parametrize(
    ("loss", "expected"),
    [
        (0, MoveClass.BEST),
        (10, MoveClass.EXCELLENT),
        (20, MoveClass.EXCELLENT),
        (21, MoveClass.GOOD),
        (50, MoveClass.GOOD),
        (51, MoveClass.INACCURACY),
        (100, MoveClass.INACCURACY),
        (101, MoveClass.MISTAKE),
        (300, MoveClass.MISTAKE),
        (301, MoveClass.BLUNDER),
        (1500, MoveClass.BLUNDER),
    ],
)
def test_thresholds(loss, expected):
    assert classify_move(loss) == expected


def test_engine_best_move_is_best_regardless_of_loss():
    assert classify_move(40, is_best=True) == MoveClass.BEST


def test_allowing_mate_is_blunder_even_with_small_loss():
    assert classify_move(30, mate_before=None, mate_after=-3) == MoveClass.BLUNDER


def test_already_getting_mated_is_not_blunder():
    # уже матовали в 2, после хода — мат в 1: оценка не изменилась (обе = -1000)
    assert classify_move(0, mate_before=-2, mate_after=-1) == MoveClass.BEST


def test_missed_mate_classification_depends_on_eval_after():
    assert classify_move(850, mate_before=2, mate_after=None, eval_after=150) == MoveClass.BLUNDER
    assert classify_move(500, mate_before=2, mate_after=None, eval_after=500) == MoveClass.MISTAKE
    assert (
        classify_move(100, mate_before=2, mate_after=None, eval_after=900) == MoveClass.INACCURACY
    )
    assert classify_move(10, mate_before=2, mate_after=None, eval_after=990) == MoveClass.INACCURACY


def test_longer_mate_is_fine_and_delivered_mate_is_fine():
    assert classify_move(0, mate_before=2, mate_after=4, eval_after=1000) == MoveClass.BEST
    assert classify_move(0, mate_before=1, mate_after=0, eval_after=1000) == MoveClass.BEST


# ------------------------------------------------------------------ точность (lichess)


def test_win_percent_known_values():
    assert win_percent(0) == pytest.approx(50.0)
    assert win_percent(1000) == pytest.approx(97.54, abs=0.01)
    assert win_percent(-1000) == pytest.approx(2.46, abs=0.01)
    assert win_percent(5000) == win_percent(1000)  # обрезка ±1000
    assert win_percent(100) == pytest.approx(59.10, abs=0.01)


def test_move_accuracy_known_values():
    assert move_accuracy(50, 50) == 100.0
    assert move_accuracy(50, 60) == 100.0
    expected = 103.1668100711649 * math.exp(-0.04354415386753951 * 10) - 3.166924740191411 + 1
    assert move_accuracy(60, 50) == pytest.approx(expected)
    assert move_accuracy(60, 50) == pytest.approx(64.58, abs=0.01)
    assert move_accuracy(97.5, 2.5) == 0.0  # обрезка снизу


def test_game_accuracy_perfect_game():
    white, black = game_accuracy([50.0] * 21)
    assert white == pytest.approx(100.0)
    assert black == pytest.approx(100.0)


def test_game_accuracy_one_side_blunders():
    series = [50.0] * 10 + [90.0] * 11  # 10-й полуход (чёрные) «уронил» партию
    white, black = game_accuracy(series)
    assert white == pytest.approx(100.0)
    assert black < 80


# ------------------------------------------------------------------ полная партия (FakeEngine)


def test_missed_mate_game(missed_mate_analysis):
    ga = missed_mate_analysis
    by_label = {m.label: m for m in ga.moves}
    qf3 = by_label["4. Qf3"]
    assert qf3.mate_before == 1
    assert qf3.classification in (MoveClass.MISTAKE, MoveClass.BLUNDER)
    assert any(m.kind == "missed_mate" and m.move_san == "Qxf7#" for m in qf3.motifs)
    nf6 = by_label["3... Nf6"]
    assert nf6.classification == MoveClass.BLUNDER
    assert any(m.kind == "allowed_mate" for m in nf6.motifs)
    last = ga.moves[-1]
    assert last.san == "Qe1#" and last.mate_after == 0
    assert last.classification == MoveClass.BEST
    assert 0 <= ga.white.accuracy <= 100 and 0 <= ga.black.accuracy <= 100
    assert ga.white.counts["blunder"] >= 1
    assert len(ga.white_evals) == len(ga.moves) + 1


def test_scholars_mate_stats(sample_analyses):
    ga = sample_analyses[0]
    assert ga.result == "1-0"
    assert ga.white.acpl == 0
    assert ga.black.counts["blunder"] == 1
    assert ga.white.accuracy > ga.black.accuracy
