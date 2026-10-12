"""Генерация демо-PGN в ``data/games/`` с проверкой легальности ходов через python-chess.

Запуск: ``uv run python scripts/make_games.py``
"""

from __future__ import annotations

from pathlib import Path

import chess
import chess.pgn

OUT = Path(__file__).resolve().parent.parent / "data" / "games"

GAMES: dict[str, list[dict]] = {
    # Короткие миниатюры с очевидными зевками (две партии в одном файле — проверка multi-game).
    "sample.pgn": [
        {
            "headers": {
                "Event": "Demo miniature (Scholar's mate)",
                "Site": "Tashkent",
                "Date": "2026.10.01",
                "Round": "1",
                "White": "Novice A",
                "Black": "Novice B",
                "Result": "1-0",
                "ECO": "C23",
            },
            "moves": "e4 e5 Bc4 Nc6 Qh5 Nf6 Qxf7#",
            "expect_mate": True,
        },
        {
            "headers": {
                "Event": "Paris (Legal's mate)",
                "Site": "Cafe de la Regence",
                "Date": "1750.??.??",
                "Round": "?",
                "White": "Legal de Kermeur",
                "Black": "Saint Brie",
                "Result": "1-0",
                "ECO": "C41",
            },
            "moves": "e4 e5 Nf3 d6 Bc4 Bg4 Nc3 g6 Nxe5 Bxd1 Bxf7+ Ke7 Nd5#",
            "expect_mate": True,
        },
    ],
    # Длинная «любительская» партия с несколькими явными зевками:
    # 13...Nh5?? (зевает коня), 15.Qe2? (оставляет коня h5), 17.Qg4?? (зевает ферзя) и др.
    "amateur.pgn": [
        {
            "headers": {
                "Event": "Club blitz (synthetic)",
                "Site": "Tashkent",
                "Date": "2026.09.14",
                "Round": "3",
                "White": "Amateur White",
                "Black": "Amateur Black",
                "Result": "0-1",
                "ECO": "C50",
                "WhiteElo": "1350",
                "BlackElo": "1410",
            },
            "moves": (
                "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 O-O O-O Re1 a6 Nbd2 Ba7 Nf1 Be6 "
                "Bxe6 fxe6 Ng3 Qd7 Be3 Bxe3 Rxe3 Nh5 Nxh5 Qf7 Qe2 Qxh5 Nd2 Qg5 Qg4 Qxg4 "
                "Nf3 Qf4 Re2 Qg4 Kf1 Rxf3 gxf3 Qxf3 Ke1 Rf8 Rd1 Qh1+ Kd2 Rxf2"
            ),
        },
    ],
    # Партия с упущенным матом: 4.Qf3? вместо 4.Qxf7#, затем 13.Qxb7?? и 13...Qe1#.
    "missed_mate.pgn": [
        {
            "headers": {
                "Event": "Online rapid (synthetic)",
                "Site": "lichess.org",
                "Date": "2026.09.20",
                "Round": "-",
                "White": "Hasty Hero",
                "Black": "Patient Pawn",
                "Result": "0-1",
                "ECO": "C23",
            },
            "moves": (
                "e4 e5 Bc4 Nc6 Qh5 Nf6 Qf3 Nd4 Qd3 d5 Bxd5 Nxd5 exd5 Qxd5 Nc3 Qxg2 "
                "Qe4 Bf5 Qxe5+ Be7 Qxc7 Nxc2+ Kd1 Qxf2 Qxb7 Qe1#"
            ),
            "expect_mate": True,
        },
    ],
}


def build_game(spec: dict) -> chess.pgn.Game:
    game = chess.pgn.Game()
    for k, v in spec["headers"].items():
        game.headers[k] = v
    node = game
    board = chess.Board()
    for san in spec["moves"].split():
        move = board.parse_san(san)  # бросит ValueError на нелегальном ходе
        board.push(move)
        node = node.add_variation(move)
    if spec.get("expect_mate") and not board.is_checkmate():
        raise ValueError(f"Ожидался мат в конце партии: {spec['headers']['Event']}")
    return game


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, specs in GAMES.items():
        text = "\n\n".join(str(build_game(s)) for s in specs) + "\n"
        (OUT / filename).write_text(text, encoding="utf-8")
        print(f"{filename}: {len(specs)} game(s)")


if __name__ == "__main__":
    main()
