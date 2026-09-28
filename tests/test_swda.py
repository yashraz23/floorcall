"""D1 and D2 construction from SwDA, on a 13-utterance conversation traced by hand.

tests/fixtures/swda/sw90utt/sw_9001_9001.utt.csv:

  0  A sd  I live in Texas. /
  1  B sd  {D Well, } I live in Ohio. /
  2  A b   Uh-huh. /
  3  B sd  And I work at a bank, {F uh, }          <- no "/": the unit continues
  4  A b   Yeah. /
  5  B +   downtown. /                             <- continuation of 3
  6  A b   Yeah, /
  7  A sd  but that's not what I asked. /          <- 6+7 is one run: "yeah, but ..."
  8  B x   <Laughter>. /
  9  A qw  So what do you do? /
 10  B aa  Oh, I see what you mean. /              <- 6-word agreement: takes the floor
 11  A sd  I was cut off, -/                       <- abandoned
 12  B sd  Sorry. /
 13  A sd  We moved here last year. /
 14  B aa  Yeah. /                                 <- 1-word agreement: minor, not a turn
 15  A sd  It was a big change. /                  <- A still holds the floor
 16  B sd  I bet. /
"""

import random
from pathlib import Path

import pytest

from floorcall.data import swda
from floorcall.questions import BACKCHANNEL, INTERRUPTION, NOISE

FIXTURE = Path(__file__).parent / "fixtures" / "swda"


@pytest.fixture(scope="module")
def conv() -> list[swda.Utt]:
    (c,) = list(swda.iter_conversations(FIXTURE, minimal_max_words=3))
    return c


def by_index(examples: list[swda.Example]) -> dict[int, list[swda.Example]]:
    out: dict[int, list[swda.Example]] = {}
    for e in examples:
        out.setdefault(int(e.id.split("-")[2]), []).append(e)
    return out


# -- text and tags ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("{D Well, } I live in Ohio. /", "Well, I live in Ohio."),
        ("And I work at a bank, {F uh, }", "And I work at a bank, uh,"),
        ("<Laughter>. /", ""),
        ("I was cut off, -/", "I was cut off,"),
        ("[ I think, + I guess ] # you know # ((maybe))", "I think, I guess you know maybe"),
        ("<<talking to child>> Okay. /", "Okay."),
        ("-- the different, - /", "the different,"),
    ],
)
def test_clean_text(raw: str, clean: str) -> None:
    assert swda.clean_text(raw) == clean


@pytest.mark.parametrize(
    ("raw", "damsl"),
    [
        ("sd", "sd"),
        ("b", "b"),
        ("qy^d", "qy^d"),  # kept whole, as in cgpotts/swda
        ("sd^e", "sd"),  # modifier dropped
        ("qr", "qy"),
        ("nn^e", "ng"),
        ("fe", "ba"),
        ("sd,sv", "sd"),  # first of several wins
        ("+", "+"),
        ("%", "%"),
    ],
)
def test_damsl_tag(raw: str, damsl: str) -> None:
    assert swda.damsl_tag(raw) == damsl


# -- D1 --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def d1(conv: list[swda.Utt]) -> dict[int, list[swda.Example]]:
    return by_index(swda.build_d1(conv, rng=random.Random(0), max_history=8))


def test_d1_finds_exactly_the_traced_examples(d1: dict[int, list[swda.Example]]) -> None:
    kinds = {i: sorted(e.kind for e in es) for i, es in d1.items()}
    assert kinds == {
        0: ["complete", "truncation"],  # B then claims the floor
        3: ["continuation"],  # no "/", A only backchannels, B resumes with "+"
        5: ["complete"],  # A claims the floor at 7; one word, so no truncation
        9: ["complete", "truncation"],  # B's six-word agreement takes the floor
        10: ["complete", "truncation"],
        12: ["complete"],
        15: ["complete", "truncation"],  # B's "I bet." takes the floor
    }
    # excluded: 1 (A only backchannels, B starts a new unit), 7 (B only laughs, A starts a new
    # unit), 11 (abandoned "-/", then B speaks: cut off, not finished), 13 (B's one-word "Yeah."
    # is a minimal response, not a turn, and A goes on: ambiguous, like 1)


def test_d1_labels(d1: dict[int, list[swda.Example]]) -> None:
    for es in d1.values():
        for e in es:
            assert e.label == ("true" if e.kind == "complete" else "false")
            assert e.decision == "turn_complete"
            assert e.event == "user_pause"
            assert e.agent_speaking is False


def test_d1_user_partial_is_the_whole_turn_so_far(d1: dict[int, list[swda.Example]]) -> None:
    (cont,) = d1[3]
    assert cont.user_partial == "Well, I live in Ohio. And I work at a bank, uh,"
    (done,) = d1[5]
    assert done.user_partial == "Well, I live in Ohio. And I work at a bank, uh, downtown."
    # turn 9 starts after B's last claim (5), and includes every word A said since, the "Yeah,"
    # at 6 too: a recogniser transcribes it like any other word
    complete9 = next(e for e in d1[9] if e.kind == "complete")
    assert complete9.user_partial == "Yeah, but that's not what I asked. So what do you do?"
    # B's one-word "Yeah." at 14 does not end A's turn
    complete15 = next(e for e in d1[15] if e.kind == "complete")
    assert complete15.user_partial == "We moved here last year. It was a big change."


def test_d1_truncation_cuts_inside_the_last_utterance(d1: dict[int, list[swda.Example]]) -> None:
    complete = next(e for e in d1[9] if e.kind == "complete")
    trunc = next(e for e in d1[9] if e.kind == "truncation")
    earlier = "Yeah, but that's not what I asked."
    assert trunc.user_partial.startswith(earlier + " So")
    last_words = trunc.user_partial[len(earlier) :].split()
    assert 1 <= len(last_words) <= 4  # "So what do you do?" has 5 words; at least one is cut
    assert complete.user_partial.startswith(trunc.user_partial)


def test_d1_roles_and_history(d1: dict[int, list[swda.Example]]) -> None:
    (done,) = d1[5]  # B is the user here; A plays the agent
    assert done.agent_last_utterance == "I live in Texas."
    assert done.recent_turns == []
    complete9 = next(e for e in d1[9] if e.kind == "complete")  # A is the user
    assert (
        complete9.agent_last_utterance
        == "Well, I live in Ohio. And I work at a bank, uh, downtown."
    )
    assert complete9.recent_turns == [{"speaker": "user", "text": "I live in Texas."}]


# -- D2 --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def d2(conv: list[swda.Utt]) -> dict[int, swda.Example]:
    es = swda.build_d2(conv, partial_words=12, max_history=8)
    out = by_index(es)
    assert all(len(v) == 1 for v in out.values())
    return {i: v[0] for i, v in out.items()}


def test_d2_finds_exactly_the_traced_examples(d2: dict[int, swda.Example]) -> None:
    assert {i: e.label for i, e in d2.items()} == {
        1: INTERRUPTION,
        2: BACKCHANNEL,
        4: BACKCHANNEL,
        6: INTERRUPTION,  # the run 6+7
        8: NOISE,
        11: INTERRUPTION,
        12: INTERRUPTION,
        13: INTERRUPTION,
        16: INTERRUPTION,
    }
    # excluded: 3 (A had only backchannelled, so B already held the floor), 5 ("+" continuation),
    # 9 (B had only laughed), 10 and 14 (aa: agreement, neither class), 15 (B had only said
    # "Yeah.", so A still held the floor: A going on is not a barge-in)


def test_d2_merges_a_run_into_one_partial(d2: dict[int, swda.Example]) -> None:
    e = d2[6]
    assert e.user_partial == "Yeah, but that's not what I asked."
    assert e.kind == "b+sd"


def test_d2_agent_is_mid_turn(d2: dict[int, swda.Example]) -> None:
    assert d2[6].agent_last_utterance == "Well, I live in Ohio. And I work at a bank, uh, downtown."
    assert d2[4].agent_last_utterance == "Well, I live in Ohio. And I work at a bank, uh,"
    assert d2[8].agent_last_utterance == "but that's not what I asked."
    assert d2[11].agent_last_utterance == "Oh, I see what you mean."
    assert all(e.agent_speaking for e in d2.values())


def test_a_short_minimal_response_does_not_take_the_floor(d2: dict[int, swda.Example]) -> None:
    # The bug this guards: SwDA tags a short "yeah" aa or ny as often as b. Counted as a turn, it
    # handed the floor to B, and A simply going on at 15 looked like A barging in on B.
    assert 15 not in d2
    assert d2[16].agent_last_utterance == "We moved here last year. It was a big change."


def test_d2_noise_can_be_empty_text(d2: dict[int, swda.Example]) -> None:
    # a laugh gives the recogniser nothing: the model sees an empty partial
    assert d2[8].user_partial == ""


def test_d2_partial_is_capped(conv: list[swda.Utt]) -> None:
    (e,) = [
        x for x in swda.build_d2(conv, partial_words=3, max_history=8) if x.id.endswith("0006-d2")
    ]
    assert e.user_partial == "Yeah, but that's"


@pytest.mark.parametrize(
    ("tags", "label"),
    [
        (["b"], BACKCHANNEL),
        (["b", "bh"], BACKCHANNEL),
        (["b", "sd"], INTERRUPTION),
        (["aa", "sv"], INTERRUPTION),
        (["x"], NOISE),
        (["x", "%"], NOISE),
        (["aa"], None),
        (["b", "aa"], None),
        (["x", "b"], None),
        ([], None),
    ],
)
def test_run_label(tags: list[str], label: str | None) -> None:
    assert swda.run_label(tags) == label


# -- hard subsets ----------------------------------------------------------------------------


def ex(label: str, text: str, split: str = "train", kind: str = "") -> swda.Example:
    return swda.Example(
        id=text,
        decision="barge_in",
        event="user_speech_during_agent",
        group="g",
        label=label,
        kind=kind,
        agent_speaking=True,
        user_partial=text,
        agent_last_utterance="",
        recent_turns=[],
        split=split,  # type: ignore[arg-type]
    )


def test_d2_hard_needs_both_classes_and_neither_dominant() -> None:
    rows = (
        [ex(BACKCHANNEL, "Yeah.") for _ in range(30)]
        + [ex(INTERRUPTION, "Yeah, but no.") for _ in range(25)]
        + [ex(BACKCHANNEL, "Uh-huh.") for _ in range(300)]
        + [ex(INTERRUPTION, "Uh-huh, and then.") for _ in range(25)]  # 8%: dominated
        + [ex(BACKCHANNEL, "Right.") for _ in range(30)]  # never opens an interruption
    )
    test_rows = [
        ex(BACKCHANNEL, "Yeah.", "test"),
        ex(INTERRUPTION, "yeah but that's not what I asked", "test"),
        ex(INTERRUPTION, "yeahs are fine", "test"),  # "yeah" only at a word boundary
        ex(BACKCHANNEL, "Uh-huh.", "test"),
        ex(NOISE, "yeah", "test"),
    ]
    shared = swda.mark_hard_d2(rows + test_rows, min_count=20, min_share=0.2)
    assert set(shared) == {"yeah"}
    assert [r.hard for r in test_rows] == [True, True, False, False, False]


def test_d2_hard_is_defined_from_train_only() -> None:
    rows = [ex(BACKCHANNEL, "Yeah.", "test") for _ in range(30)] + [
        ex(INTERRUPTION, "Yeah, but no.", "test") for _ in range(30)
    ]
    assert swda.mark_hard_d2(rows, min_count=20, min_share=0.2) == {}
    assert not any(r.hard for r in rows)


def test_d1_hard_marks_truncations_ending_like_a_turn() -> None:
    def d1ex(kind: str, text: str, split: str = "train") -> swda.Example:
        e = ex("true" if kind == "complete" else "false", text, split, kind)
        e.decision = "turn_complete"
        return e

    rows = (
        [d1ex("complete", "I live in Texas.") for _ in range(12)]
        + [d1ex("truncation", "I live in Texas") for _ in range(10)]  # r = 1.2
        + [d1ex("truncation", "I went to the") for _ in range(10)]  # r = 0
    )
    tests = [
        d1ex("truncation", "we moved to Texas", "test"),
        d1ex("truncation", "put it on the", "test"),
    ]
    turn_final = swda.mark_hard_d1(rows + tests, min_count=10)
    assert set(turn_final) == {"texas"}
    assert [t.hard for t in tests] == [True, False]
