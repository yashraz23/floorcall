"""The Space's views (D-050), from the committed replay run and results, without Gradio."""

import json
import re
from typing import Any

import pytest

from floorcall import space
from floorcall.config import REPO_ROOT

REPLAY = json.loads((REPO_ROOT / "results" / "replay" / "replay.json").read_text(encoding="utf-8"))
SCRIPTS = {
    p.stem: json.loads(p.read_text(encoding="utf-8"))
    for p in (REPO_ROOT / "demo" / "scripts").glob("*.json")
}


def test_scores_match_the_committed_replay_report() -> None:
    text = (REPO_ROOT / "results" / "replay" / "replay.txt").read_text(encoding="utf-8")
    for agent in ("naive", "floorcall"):
        ok, points = space.score(REPLAY, agent)
        assert f"{agent:9s} acted as wanted at {ok} of {points}" in text


@pytest.mark.parametrize("result", REPLAY["scripts"], ids=lambda r: r["id"])
def test_every_script_renders_every_decision_point(result: dict[str, Any]) -> None:
    script = SCRIPTS[result["id"]]
    html = space.replay_html(script, result, REPLAY)
    assert html.count("<svg") == 1 and html.count("</svg>") == 1
    rows = re.findall(r'<tr><td><span class="fc-n">(\d+)</span>', html)
    outcomes = {
        (o["seg"], o["event"])
        for a in ("naive", "floorcall")
        for o in result["runs"][a]["outcomes"]
    }
    missed = {
        (m["seg"], "user_speech_during_agent")
        for a in ("naive", "floorcall")
        for m in result["runs"][a]["missed"]
    }
    assert [int(n) for n in rows] == list(range(1, len(outcomes | missed) + 1))
    wrong = sum(
        o["ok"] is False for a in ("naive", "floorcall") for o in result["runs"][a]["outcomes"]
    )
    assert html.count("✗") >= wrong


def test_a_stopped_line_is_drawn_unsaid_from_the_stop() -> None:
    result = next(r for r in REPLAY["scripts"] if r["id"] == "01_backchannels")
    svg = space.timeline_svg(SCRIPTS["01_backchannels"], result)
    # naive stops for "uh-huh"; floorcall keeps talking, so exactly one unsaid stretch
    assert svg.count('fill="url(#fc-unsaid)"') == 1


def test_text_is_escaped() -> None:
    script = {**SCRIPTS["01_backchannels"], "title": "<script>x</script>"}
    result = next(r for r in REPLAY["scripts"] if r["id"] == "01_backchannels")
    assert "<script>x" not in space.replay_html(script, result, REPLAY)


def test_try_it_shows_the_action_the_bars_and_the_state() -> None:
    event = {
        "action": "keep_talking",
        "route": None,
        "escalate": False,
        "reason": "backchannel (p=1.00)",
        "probabilities": {"barge_in": {"backchannel": 0.9, "interruption": 0.07, "noise": 0.03}},
        "compute_ms": 812.4,
        "device": "cpu (fp32)",
        "state_tokens": 31,
    }
    html = space.try_it_html(event, {"user_partial": "uh huh"})
    assert "keep talking" in html and "0.900" in html and "812 ms" in html and "uh huh" in html


def test_results_render_every_table_without_todo() -> None:
    md = space.results_markdown()
    for heading in ("Table A", "Table B", "Table C", "Table D", "Operating points"):
        assert heading in md
    assert "TODO" not in md


def test_markdown_tables_headings_and_links_render() -> None:
    md = "### Table A\n\n| a | b |\n|---|---|\n| **1** | [x](https://e.org) |\n\nplain <text> here"
    out = space.markdown_html(md)
    assert "<h3>Table A</h3>" in out
    assert "<th>a</th>" in out and "<td><b>1</b></td>" in out
    assert '<a href="https://e.org">x</a>' in out
    assert "&lt;text&gt;" in out and "---" not in out


def test_the_static_page_has_every_script_and_every_table() -> None:
    from html.parser import HTMLParser

    page = space.static_page(
        REPLAY, SCRIPTS, space.results_markdown(), [("f.png", "a figure")], "o/m", "abcdef1234"
    )
    assert page.count('class="fc-script"') == len(REPLAY["scripts"])
    assert page.count("<svg") == len(REPLAY["scripts"])
    for heading in ("Table A", "Table B", "Table C", "Table D"):
        assert heading in page
    assert "TODO" not in page and 'src="figures/f.png"' in page

    class Balance(HTMLParser):
        VOID = frozenset({"meta", "img", "br", "circle", "rect", "line", "input", "path"})

        def __init__(self) -> None:
            super().__init__()
            self.stack: list[str] = []

        def handle_starttag(self, tag: str, attrs: object) -> None:
            if tag not in self.VOID:
                self.stack.append(tag)

        def handle_startendtag(self, tag: str, attrs: object) -> None:
            pass

        def handle_endtag(self, tag: str) -> None:
            if tag in self.VOID:
                return
            assert self.stack and self.stack[-1] == tag, (tag, self.stack[-3:])
            self.stack.pop()

    b = Balance()
    b.feed(page)
    assert b.stack == []
