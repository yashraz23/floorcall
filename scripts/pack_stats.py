"""How much context the token budget costs: pack every frozen test row with the real tokenizer.

    uv run python scripts/pack_stats.py

Writes results/pack_stats.json. For each test set: the budget, the distribution of packed state
sizes, and how often the packer had to drop history, trim the agent's utterance, or (never, one
hopes) trim the user's own words.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from floorcall import questions
from floorcall.config import LayaSettings, get_settings
from floorcall.data.build import test_file
from floorcall.data.freeze import read_jsonl_gz
from floorcall.model.laya_adapter import LayaDecider
from floorcall.state import Snapshot, Turn, pack_state

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pack_stats.json"


def snapshot(row: dict[str, Any]) -> Snapshot:
    s = row["snapshot"]
    return Snapshot(
        agent_speaking=s["agent_speaking"],
        user_partial=s["user_partial"],
        agent_last_utterance=s["agent_last_utterance"],
        recent_turns=tuple(Turn(t["speaker"], t["text"]) for t in s["recent_turns"]),
    )


def main() -> None:
    cfg = get_settings()
    d = LayaDecider(LayaSettings(**{**cfg.laya.model_dump(), "device": "cpu"}))
    out: dict[str, Any] = {}
    for decision, event in (
        ("turn_complete", questions.Event.USER_PAUSE),
        ("barge_in", questions.Event.USER_SPEECH_DURING_AGENT),
        ("route", questions.Event.USER_PAUSE),
    ):
        rows = read_jsonl_gz(cfg.paths.test_frozen / test_file(decision))
        room = d.state_room(questions.questions_for(event))
        budget = room - cfg.state.safety_margin_tokens
        packed = [pack_state(snapshot(r), budget=budget, count_tokens=d.count_tokens) for r in rows]
        tokens = np.array([p.n_tokens for p in packed])
        out[decision] = {
            "event": event.value,
            "room": room,
            "budget": budget,
            "rows": len(rows),
            "tokens": {q: int(np.percentile(tokens, q)) for q in (50, 90, 99, 100)},
            "share_with_turns_dropped": float(np.mean([p.turns_dropped > 0 for p in packed])),
            "share_with_agent_trimmed": float(np.mean([p.agent_words_dropped > 0 for p in packed])),
            "share_with_user_trimmed": float(np.mean([p.user_words_dropped > 0 for p in packed])),
            "mean_turns_kept": float(np.mean([p.turns_kept for p in packed])),
        }
        print(decision, json.dumps(out[decision]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
