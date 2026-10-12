import chess
import pytest

from chess_analyzer.pgn import PGNError, count_plies, game_headers, iter_positions, load_games

TWO_GAMES = """[Event "A"]
[White "W1"]
[Black "B1"]
[Result "1-0"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0

[Event "B"]
[White "W2"]
[Black "B2"]
[Result "0-1"]

1. f3 e5 2. g4 Qh4# 0-1
"""


def test_load_multiple_games_from_string():
    games = load_games(TWO_GAMES)
    assert len(games) == 2
    assert game_headers(games[0])["White"] == "W1"
    assert game_headers(games[1])["Result"] == "0-1"
    assert count_plies(games[1]) == 4


def test_load_from_file(data_dir):
    games = load_games(data_dir / "sample.pgn")
    assert len(games) == 2
    assert games[1].headers["White"] == "Legal de Kermeur"
    assert len(load_games(str(data_dir / "amateur.pgn"))) == 1  # строковый путь тоже работает


def test_all_demo_games_are_legal(data_dir):
    files = sorted(data_dir.glob("*.pgn"))
    assert len(files) == 3
    for path in files:
        for game in load_games(path):
            assert not game.errors
            assert count_plies(game) > 0


def test_iter_positions():
    game = load_games(TWO_GAMES)[1]
    steps = list(iter_positions(game))
    assert [s.san for s in steps] == ["f3", "e5", "g4", "Qh4#"]
    assert steps[0].color == chess.WHITE and steps[1].color == chess.BLACK
    assert steps[3].move_number == 2
    assert steps[0].board_before.fen() == chess.STARTING_FEN
    assert steps[3].board_after.is_checkmate()


def test_illegal_move_raises():
    with pytest.raises(PGNError):
        load_games('[Event "x"]\n\n1. e4 e5 2. Ke3 *\n')


def test_missing_file_raises():
    with pytest.raises(PGNError):
        load_games("no_such_file.pgn")
