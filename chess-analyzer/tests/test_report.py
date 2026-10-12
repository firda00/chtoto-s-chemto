import json

from chess_analyzer.explain import explain_game
from chess_analyzer.llm import MockLLM
from chess_analyzer.report import (
    lichess_url,
    render_html,
    render_markdown,
    write_report,
)


def _explained(analysis):
    ga = analysis.model_copy(deep=True)
    return explain_game(ga, MockLLM(), lang="ru")


def test_lichess_url():
    url = lichess_url("8/8/8/8/8/8/8/K6k w - - 0 1")
    assert url == "https://lichess.org/analysis/standard/8/8/8/8/8/8/8/K6k_w_-_-_0_1"


def test_markdown_sections(missed_mate_analysis):
    md = render_markdown([_explained(missed_mate_analysis)], lang="ru")
    assert "# Анализ шахматных партий" in md
    assert "### Сводка" in md and "### Критические моменты" in md
    assert "| Hasty Hero | белые |" in md
    assert "#### 4. Qf3? — ошибка" in md or "#### 4. Qf3?? — грубая ошибка" in md
    assert "https://lichess.org/analysis/standard/" in md
    assert "![4. Qf3](boards/g1_ply7.svg)" in md
    assert "> [mock]" in md
    # таблица Markdown: заголовок и разделитель на соседних строках
    assert "|---|---|---|---|" in md


def test_html_has_svg_boards_and_table(missed_mate_analysis, sample_analyses):
    html = render_html([_explained(missed_mate_analysis), *sample_analyses], lang="en")
    assert html.startswith("<!doctype html>")
    assert '<table class="summary">' in html
    assert "Critical moments" in html
    assert html.count("<svg") >= 3  # доски + графики
    assert 'class="graph"' in html
    assert 'id="g1_ply7"' in html
    assert "Legal de Kermeur" in html


def test_write_report_files(tmp_path, sample_analyses):
    paths = write_report(sample_analyses, tmp_path, lang="ru")
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
    boards = list((tmp_path / "boards").glob("*.svg"))
    assert boards and boards[0].read_text(encoding="utf-8").startswith("<svg")
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert len(data) == 2 and data[0]["white"]["name"] == "Novice A"
