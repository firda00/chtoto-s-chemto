from chess_analyzer.analysis import MoveClass
from chess_analyzer.explain import build_prompt, explain_game, explain_move, template_explain
from chess_analyzer.llm import MockLLM, get_llm


def _qf3(analysis):
    return next(m for m in analysis.moves if m.label == "4. Qf3")


def test_prompt_contains_key_facts(missed_mate_analysis):
    move = _qf3(missed_mate_analysis)
    system, prompt = build_prompt(move, "ru")
    assert "шахматный тренер" in system
    assert f"FEN: {move.fen_before}" in prompt
    assert "Played move: 4. Qf3" in prompt
    assert "Best move: Qxf7#" in prompt
    assert "Eval swing:" in prompt
    assert "missed a mate in 1" in prompt
    assert "Language: Russian" in prompt


def test_mock_llm_explanation_ru_and_en(missed_mate_analysis):
    move = _qf3(missed_mate_analysis)
    llm = MockLLM()
    ru = explain_move(move, llm, "ru")
    en = explain_move(move, llm, "en")
    assert ru.startswith("[mock] Ход 4. Qf3") and "Qxf7#" in ru
    assert en.startswith("[mock] The move 4. Qf3") and "Qxf7#" in en
    assert len(llm.calls) == 2


def test_template_explain_without_llm(missed_mate_analysis):
    move = _qf3(missed_mate_analysis)
    ru = template_explain(move, "ru")
    assert "4. Qf3" in ru and "упущен мат в 1" in ru.lower() and "Qxf7#" in ru
    en = template_explain(move, "en")
    assert "Missed a mate in 1" in en and "Better was Qxf7#" in en
    assert explain_move(move, None, "ru") == ru


def test_llm_failure_falls_back_to_template(missed_mate_analysis):
    move = _qf3(missed_mate_analysis)

    def boom(prompt, system):
        raise ConnectionError("offline")

    assert explain_move(move, MockLLM(responder=boom), "en") == template_explain(move, "en")
    assert explain_move(move, MockLLM(responder=lambda p, s: "  "), "en") == template_explain(
        move, "en"
    )


def test_explain_game_fills_only_critical(missed_mate_analysis):
    ga = missed_mate_analysis.model_copy(deep=True)
    explain_game(ga, MockLLM(), lang="ru", min_class=MoveClass.MISTAKE)
    critical = ga.critical_moves(MoveClass.MISTAKE)
    assert critical and all(m.explanation for m in critical)
    assert all(m.explanation is None for m in ga.moves if m not in critical)


def test_get_llm_auto_without_key_is_none():
    assert get_llm("auto") is None
    assert isinstance(get_llm("mock"), MockLLM)
