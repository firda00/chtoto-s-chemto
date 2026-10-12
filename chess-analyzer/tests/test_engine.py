import os

import chess
import httpx
import pytest

from chess_analyzer.engine import (
    ApiEngine,
    EngineEval,
    FakeEngine,
    make_engine,
    static_eval,
    terminal_eval,
)

BACK_RANK = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"


def test_static_eval_symmetric_start():
    assert static_eval(chess.Board()) == 0


def test_fake_engine_is_deterministic():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    a, b = FakeEngine().analyse(board), FakeEngine().analyse(board)
    assert a == b
    assert a.best_move is not None
    assert chess.Move.from_uci(a.best_move) in board.legal_moves
    assert a.pv[0] == a.best_move


def test_fake_engine_prefers_free_capture():
    board = chess.Board("4k3/8/8/8/3q4/4P3/8/4K3 w - - 0 1")  # ферзь d4 висит под пешкой e3
    ev = FakeEngine().analyse(board)
    assert ev.best_move == "e3d4"
    assert ev.score_cp > 0


def test_fake_engine_finds_mate_in_one():
    ev = FakeEngine().analyse(chess.Board(BACK_RANK))
    assert ev.best_move == "a1a8"
    assert ev.mate == 1
    assert ev.pov_mate(chess.BLACK) == -1


def test_terminal_eval_checkmate():
    board = chess.Board()
    for san in ["f3", "e5", "g4", "Qh4#"]:
        board.push_san(san)
    ev = terminal_eval(board)
    assert ev is not None and ev.pov_cp(chess.BLACK) == 1000
    assert FakeEngine().analyse(board) == ev


def test_pov_cp_caps_and_flips():
    ev = EngineEval(score_cp=2500)
    assert ev.pov_cp(chess.WHITE) == 1000
    assert ev.pov_cp(chess.BLACK) == -1000
    assert EngineEval(mate=-3).pov_cp(chess.WHITE) == -1000


def test_api_engine_parses_stockfish_online_response():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.url.params)
        return httpx.Response(
            200,
            json={
                "success": True,
                "evaluation": 0.36,
                "mate": None,
                "bestmove": "bestmove e2e4 ponder e7e5",
                "continuation": "e2e4 e7e5 g1f3",
            },
        )

    engine = ApiEngine(
        url="http://test/api", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    ev = engine.analyse(chess.Board(), depth=20)
    assert ev.score_cp == 36
    assert ev.best_move == "e2e4"
    assert ev.pv == ["e2e4", "e7e5", "g1f3"]
    assert seen["depth"] == "15"  # ограничено MAX_DEPTH
    assert seen["fen"] == chess.STARTING_FEN


def test_api_engine_mate_and_error():
    def handler(request):
        if request.url.params["depth"] == "1":
            return httpx.Response(200, json={"success": False, "data": "bad fen"})
        return httpx.Response(
            200, json={"success": True, "evaluation": None, "mate": -2, "bestmove": "bestmove a1a2"}
        )

    engine = ApiEngine(url="http://t", client=httpx.Client(transport=httpx.MockTransport(handler)))
    ev = engine.analyse(chess.Board())
    assert ev.mate == -2 and ev.score_cp is None and ev.pv == ["a1a2"]
    with pytest.raises(RuntimeError):
        engine.analyse(chess.Board(), depth=1)


def test_api_engine_url_from_env(monkeypatch):
    monkeypatch.setenv("STOCKFISH_API_URL", "http://my-stockfish/api")
    assert ApiEngine().url == "http://my-stockfish/api"


def test_make_engine():
    assert isinstance(make_engine("fake"), FakeEngine)
    with pytest.raises(ValueError):
        make_engine("nope")


@pytest.mark.stockfish
@pytest.mark.skipif(not os.environ.get("STOCKFISH_PATH"), reason="STOCKFISH_PATH не задан")
def test_real_stockfish_finds_mate():
    from chess_analyzer.engine import StockfishEngine

    engine = StockfishEngine()
    try:
        ev = engine.analyse(chess.Board(BACK_RANK), depth=10)
        assert ev.mate == 1 and ev.best_move == "a1a8"
    finally:
        engine.close()
