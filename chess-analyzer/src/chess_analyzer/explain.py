"""Человеческие объяснения ошибок: через LLM (``llm.py``) или шаблоном без LLM."""

from __future__ import annotations

import logging

from .analysis import GameAnalysis, MoveAnalysis, MoveClass
from .llm import LLM
from .motifs import Motif

log = logging.getLogger(__name__)

PIECES_RU = {
    "pawn": "пешка",
    "knight": "конь",
    "bishop": "слон",
    "rook": "ладья",
    "queen": "ферзь",
    "king": "король",
}

CLASS_NAMES = {
    "ru": {
        MoveClass.BEST: "лучший ход",
        MoveClass.EXCELLENT: "отличный ход",
        MoveClass.GOOD: "хороший ход",
        MoveClass.INACCURACY: "неточность",
        MoveClass.MISTAKE: "ошибка",
        MoveClass.BLUNDER: "грубая ошибка (зевок)",
    },
    "en": {
        MoveClass.BEST: "best move",
        MoveClass.EXCELLENT: "excellent move",
        MoveClass.GOOD: "good move",
        MoveClass.INACCURACY: "inaccuracy",
        MoveClass.MISTAKE: "mistake",
        MoveClass.BLUNDER: "blunder",
    },
}

SYSTEM_PROMPT = {
    "ru": (
        "Ты — шахматный тренер. Объясняй ошибки просто и конкретно, для игрока-любителя. "
        "Опирайся только на переданные факты (оценки движка, лучший ход, линию, мотив), "
        "не выдумывай варианты. Ответ: 2–3 предложения на русском, без markdown."
    ),
    "en": (
        "You are a chess coach. Explain mistakes simply and concretely to an amateur player. "
        "Use only the facts provided (engine evals, best move, line, motif); do not invent "
        "variations. Answer in 2-3 sentences in English, no markdown."
    ),
}


def format_eval(cp: int, mate: int | None = None) -> str:
    """``+1.50`` / ``-0.30`` / ``#3`` / ``#-2`` (с точки зрения ходившего)."""
    if mate is not None:
        return "#" if mate == 0 else f"#{mate}"
    return f"{cp / 100:+.2f}"


def _squares(m: Motif) -> str:
    return ", ".join(m.squares)


def describe_motif(m: Motif, lang: str = "ru") -> str:
    """Короткое описание мотива на ``lang`` (ru/en)."""
    piece = m.piece or "piece"
    move = m.move_san or "?"
    if lang == "ru":
        p = PIECES_RU.get(piece, "фигура")
        return {
            "hanging_piece": f"{p} на {_squares(m)} остаётся под боем без достаточной защиты",
            "missed_mate": f"упущен мат в {m.mate_in} ({move})",
            "allowed_mate": f"сопернику разрешён мат в {m.mate_in} ({move})",
            "missed_fork": f"упущена вилка {move} с атакой на {_squares(m)}",
            "allowed_fork": f"соперник получает вилку {move} с атакой на {_squares(m)}",
            "missed_pin": f"упущена связка {move} ({p} на {_squares(m)} связан с королём)",
            "allowed_pin": f"соперник связывает фигуру ходом {move} ({p} на {_squares(m)})",
            "missed_capture": f"упущено взятие: {p} на {_squares(m)} ({move})",
        }[m.kind]
    return {
        "hanging_piece": f"the {piece} on {_squares(m)} is left hanging",
        "missed_mate": f"missed a mate in {m.mate_in} ({move})",
        "allowed_mate": f"allowed the opponent a mate in {m.mate_in} ({move})",
        "missed_fork": f"missed the fork {move} hitting {_squares(m)}",
        "allowed_fork": f"allowed the fork {move} hitting {_squares(m)}",
        "missed_pin": f"missed the pin {move} (the {piece} on {_squares(m)} is pinned)",
        "allowed_pin": f"allowed the pin {move} (the {piece} on {_squares(m)})",
        "missed_capture": f"missed winning the {piece} on {_squares(m)} ({move})",
    }[m.kind]


def build_prompt(move: MoveAnalysis, lang: str = "ru") -> tuple[str, str]:
    """(system, prompt) для LLM. Поля ``Key: value`` — по одному на строку."""
    lang = "ru" if lang.startswith("ru") else "en"
    motifs = "; ".join(describe_motif(m, "en") for m in move.motifs) or "none detected"
    side = "White" if move.color == "white" else "Black"
    prompt = "\n".join(
        [
            "Explain why the played move is a mistake and what the better plan was.",
            f"Language: {'Russian' if lang == 'ru' else 'English'}",
            f"FEN: {move.fen_before}",
            f"Side to move: {side}",
            f"Played move: {move.label}",
            f"Best move: {move.best_move_san or 'unknown'}",
            f"Best line: {' '.join(move.best_line_san) or 'unknown'}",
            f"Eval before: {format_eval(move.eval_before, move.mate_before)}",
            f"Eval after: {format_eval(move.eval_after, move.mate_after)}",
            f"Eval swing: {move.eval_swing / 100:+.2f} pawns (cp loss {move.cp_loss})",
            f"Classification: {move.classification.value}",
            f"Motif: {motifs}",
            "Evaluations are from the point of view of the side that moved.",
        ]
    )
    return SYSTEM_PROMPT[lang], prompt


def template_explain(move: MoveAnalysis, lang: str = "ru") -> str:
    """Объяснение без LLM — по шаблону из фактов анализа."""
    lang = "ru" if lang.startswith("ru") else "en"
    cls = CLASS_NAMES[lang][move.classification]
    before = format_eval(move.eval_before, move.mate_before)
    after = format_eval(move.eval_after, move.mate_after)
    motif_text = "; ".join(describe_motif(m, lang) for m in move.motifs)
    line = " ".join(move.best_line_san[:6])
    if lang == "ru":
        parts = [
            f"{move.label} — {cls}: оценка изменилась с {before} до {after} "
            f"(потеря {move.cp_loss} сп)."
        ]
        if motif_text:
            parts.append(motif_text[0].upper() + motif_text[1:] + ".")
        if move.best_move_san:
            parts.append(
                f"Сильнее было {move.best_move_san}" + (f" (линия: {line})." if line else ".")
            )
    else:
        parts = [
            f"{move.label} is a {cls}: the evaluation dropped from {before} to {after} "
            f"({move.cp_loss} cp lost)."
        ]
        if motif_text:
            parts.append(motif_text[0].upper() + motif_text[1:] + ".")
        if move.best_move_san:
            parts.append(
                f"Better was {move.best_move_san}" + (f" (line: {line})." if line else ".")
            )
    return " ".join(parts)


def explain_move(move: MoveAnalysis, llm: LLM | None = None, lang: str = "ru") -> str:
    """Объяснение через LLM; при отсутствии LLM, ошибке или пустом ответе — шаблон."""
    if llm is None:
        return template_explain(move, lang)
    system, prompt = build_prompt(move, lang)
    try:
        result = llm.complete(prompt, system=system)
    except Exception as exc:  # сеть/ключ/лимиты — не роняем отчёт
        log.warning("LLM недоступен (%s), используем шаблон", exc)
        return template_explain(move, lang)
    text = result.get("explanation", "") if isinstance(result, dict) else str(result)
    return text.strip() or template_explain(move, lang)


def explain_game(
    analysis: GameAnalysis,
    llm: LLM | None = None,
    lang: str = "ru",
    min_class: MoveClass = MoveClass.MISTAKE,
) -> GameAnalysis:
    """Заполнить ``explanation`` для всех ходов не лучше ``min_class``."""
    for move in analysis.critical_moves(min_class):
        move.explanation = explain_move(move, llm, lang)
    return analysis
