"""Отчёты: Markdown + HTML (jinja2) + JSON, SVG-доски критических позиций (``chess.svg``)."""

from __future__ import annotations

import json
from pathlib import Path

import chess
import chess.svg
from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from .analysis import GameAnalysis, MoveAnalysis, MoveClass
from .explain import CLASS_NAMES, PIECES_RU, describe_motif, format_eval

TEMPLATES = Path(__file__).parent / "templates"

NAG = {
    MoveClass.INACCURACY: "?!",
    MoveClass.MISTAKE: "?",
    MoveClass.BLUNDER: "??",
}

LABELS = {
    "ru": {
        "title": "Анализ шахматных партий",
        "game": "Партия",
        "player": "Игрок",
        "color": "Цвет",
        "white": "белые",
        "black": "чёрные",
        "accuracy": "Точность",
        "acpl": "ACPL",
        "result": "Результат",
        "engine": "Движок",
        "depth": "глубина",
        "summary": "Сводка",
        "critical": "Критические моменты",
        "no_critical": "Ошибок уровня «ошибка» и хуже не найдено.",
        "played": "Сыграно",
        "best": "Лучше",
        "line": "Линия",
        "eval": "Оценка (для ходившего)",
        "loss": "Потеря",
        "motifs": "Мотивы",
        "explanation": "Объяснение",
        "lichess": "Открыть на lichess",
        "graph": "График оценки (белые сверху)",
        "moves": "Ходов",
    },
    "en": {
        "title": "Chess game analysis",
        "game": "Game",
        "player": "Player",
        "color": "Color",
        "white": "white",
        "black": "black",
        "accuracy": "Accuracy",
        "acpl": "ACPL",
        "result": "Result",
        "engine": "Engine",
        "depth": "depth",
        "summary": "Summary",
        "critical": "Critical moments",
        "no_critical": "No mistakes or blunders found.",
        "played": "Played",
        "best": "Best",
        "line": "Line",
        "eval": "Eval (mover's view)",
        "loss": "Loss",
        "motifs": "Motifs",
        "explanation": "Explanation",
        "lichess": "Open on lichess",
        "graph": "Evaluation graph (White up)",
        "moves": "Moves",
    },
}


def lichess_url(fen: str) -> str:
    """Ссылка на доску анализа lichess для FEN."""
    return "https://lichess.org/analysis/standard/" + fen.replace(" ", "_")


def board_svg(move: MoveAnalysis, size: int = 360) -> str:
    """SVG доски *до* хода: красная стрелка — сыгранный ход, зелёная — лучший."""
    board = chess.Board(move.fen_before)
    played = chess.Move.from_uci(move.uci)
    arrows = [chess.svg.Arrow(played.from_square, played.to_square, color="#d9534f")]
    if move.best_move_uci and move.best_move_uci != move.uci:
        best = chess.Move.from_uci(move.best_move_uci)
        arrows.append(chess.svg.Arrow(best.from_square, best.to_square, color="#2e9e4f"))
    orientation = chess.WHITE if move.color == "white" else chess.BLACK
    check = board.king(board.turn) if board.is_check() else None
    return chess.svg.board(board, orientation=orientation, arrows=arrows, size=size, check=check)


def eval_graph_svg(
    evals: list[int], marks: list[int] | None = None, width: int = 640, height: int = 140
) -> str:
    """Простой график оценки белых по позициям (обрезано до ±1000 сп)."""
    n = max(1, len(evals) - 1)

    def xy(i: int, cp: int) -> tuple[float, float]:
        x = i * width / n
        y = height / 2 - max(-1000, min(1000, cp)) / 1000 * (height / 2 - 6)
        return round(x, 1), round(y, 1)

    points = " ".join(f"{x},{y}" for x, y in (xy(i, cp) for i, cp in enumerate(evals)))
    dots = ""
    for ply in marks or []:
        if 0 <= ply < len(evals):
            x, y = xy(ply, evals[ply])
            dots += f'<circle cx="{x}" cy="{y}" r="4" fill="#d9534f"/>'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" height="{height}" class="graph">'
        f'<rect width="{width}" height="{height / 2}" fill="#f4f1ea"/>'
        f'<rect y="{height / 2}" width="{width}" height="{height / 2}" fill="#3b3a36"/>'
        f'<line x1="0" y1="{height / 2}" x2="{width}" y2="{height / 2}" stroke="#999"/>'
        f'<polyline fill="none" stroke="#e0a526" stroke-width="2" points="{points}"/>'
        f"{dots}</svg>"
    )


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["fmt_eval"] = lambda cp, mate=None: format_eval(cp, mate)
    env.filters["lichess"] = lichess_url
    return env


def _context(analyses: list[GameAnalysis], lang: str, min_class: MoveClass) -> dict:
    lang = "ru" if lang.startswith("ru") else "en"
    games = []
    for gi, ga in enumerate(analyses, start=1):
        crit = []
        for m in ga.critical_moves(min_class):
            crit.append(
                {
                    "m": m,
                    "id": f"g{gi}_ply{m.ply}",
                    "nag": NAG.get(m.classification, ""),
                    "cls": CLASS_NAMES[lang][m.classification],
                    "motifs": [describe_motif(x, lang) for x in m.motifs],
                    "svg": Markup(board_svg(m)),
                    "url": lichess_url(m.fen_before),
                }
            )
        games.append(
            {
                "n": gi,
                "ga": ga,
                "critical": crit,
                "graph": Markup(eval_graph_svg(ga.white_evals, [c["m"].ply for c in crit])),
                "players": [ga.white, ga.black],
            }
        )
    return {
        "t": LABELS[lang],
        "lang": lang,
        "games": games,
        "class_names": CLASS_NAMES[lang],
        "classes": list(MoveClass),
        "pieces_ru": PIECES_RU,
    }


def render_markdown(
    analyses: list[GameAnalysis],
    lang: str = "ru",
    min_class: MoveClass = MoveClass.MISTAKE,
    boards_dir: str = "boards",
) -> str:
    ctx = _context(analyses, lang, min_class)
    ctx["boards_dir"] = boards_dir
    return _env().get_template("report.md.j2").render(**ctx)


def render_html(
    analyses: list[GameAnalysis], lang: str = "ru", min_class: MoveClass = MoveClass.MISTAKE
) -> str:
    return _env().get_template("report.html.j2").render(**_context(analyses, lang, min_class))


def to_json(analyses: list[GameAnalysis]) -> list[dict]:
    return [ga.model_dump(mode="json") for ga in analyses]


def write_report(
    analyses: list[GameAnalysis],
    out_dir: str | Path,
    lang: str = "ru",
    min_class: MoveClass = MoveClass.MISTAKE,
) -> dict[str, Path]:
    """Записать ``report.md``, ``report.html``, ``report.json`` и ``boards/*.svg``."""
    out = Path(out_dir)
    boards = out / "boards"
    boards.mkdir(parents=True, exist_ok=True)
    for gi, ga in enumerate(analyses, start=1):
        for m in ga.critical_moves(min_class):
            (boards / f"g{gi}_ply{m.ply}.svg").write_text(board_svg(m), encoding="utf-8")
    paths = {
        "markdown": out / "report.md",
        "html": out / "report.html",
        "json": out / "report.json",
    }
    paths["markdown"].write_text(render_markdown(analyses, lang, min_class), encoding="utf-8")
    paths["html"].write_text(render_html(analyses, lang, min_class), encoding="utf-8")
    paths["json"].write_text(
        json.dumps(to_json(analyses), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths
