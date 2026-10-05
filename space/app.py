"""floorcall's Hugging Face Space: Replay, Try it, Results (DECISIONS.md D-044, D-050).

Staged by `uv run floorcall release space`, which copies this file, the committed replay run, the
hand-written replay scripts, three threshold files and the rendered results into one folder. The
views come from floorcall.space; this file only wires them to Gradio. No API key is needed, and
nothing here contains real Twitter text.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import gradio as gr

from floorcall import space

DATA = Path(__file__).parent / "data"
CONFIG = json.loads((DATA / "space.json").read_text(encoding="utf-8"))
REPLAY = json.loads((DATA / "replay.json").read_text(encoding="utf-8"))
SCRIPTS = {
    p.stem: json.loads(p.read_text(encoding="utf-8"))
    for p in sorted((DATA / "scripts").glob("*.json"))
}
RESULTS = {s["id"]: s for s in REPLAY["scripts"]}
CHOICES = [(f"{sid[:2]} · {SCRIPTS[sid]['title']}", sid) for sid in RESULTS]

_lock = threading.Lock()
_processor: Any = None


def processor() -> Any:
    """Load the released checkpoint once, on CPU, with the thresholds replay serves."""
    global _processor
    with _lock:
        if _processor is None:
            from huggingface_hub import snapshot_download

            from floorcall.config import LayaSettings, get_settings
            from floorcall.decider import Decider
            from floorcall.pipeline.operating import served_policy
            from floorcall.pipeline.processor import DecisionProcessor

            path = snapshot_download(CONFIG["model_id"], revision=CONFIG["model_revision"])
            settings = get_settings()
            laya = LayaSettings(checkpoint=path, revision=None, device="cpu")
            settings = settings.model_copy(update={"laya": laya})
            served = served_policy(settings, results=DATA / "results")
            _processor = DecisionProcessor(
                Decider.load(settings), served.policy, device="cpu (fp32)"
            )
    return _processor


def show_script(script_id: str) -> str:
    return space.replay_html(SCRIPTS[script_id], RESULTS[script_id], REPLAY)


def parse_turns(text: str) -> tuple[Any, ...]:
    from floorcall.state import Turn

    turns = []
    for line in text.splitlines():
        who, _, said = line.partition(":")
        who = who.strip().lower()
        if said.strip() and who in ("user", "agent"):
            turns.append(Turn(who, said.strip()))  # type: ignore[arg-type]
    return tuple(turns)


def decide(event: str, agent_last: str, user_words: str, history: str, silence_ms: float) -> str:
    from floorcall.questions import Event
    from floorcall.state import Snapshot

    if not user_words.strip():
        return "<i>Type what the user said.</i>"
    p = processor()
    speaking = event == "user_speech_during_agent"
    snap = Snapshot(
        agent_speaking=speaking,
        user_partial=user_words,
        agent_last_utterance=agent_last,
        recent_turns=parse_turns(history),
    )
    if speaking:
        ev = p.on_user_speech_during_agent(snap)
    else:
        ev = p.on_user_pause(snap, silence_ms=float(silence_ms))
    packed = p.decider.pack(Event(event), snap).state
    return space.try_it_html(ev.__dict__ | {"event": ev.event.value}, packed)


EXAMPLES = [
    [
        "user_speech_during_agent",
        "your card was declined because the store ran it as an international",
        "uh-huh",
        "user: my card got declined this morning",
        300,
    ],
    [
        "user_speech_during_agent",
        "your card was declined because the store ran it as an international",
        "yeah but why was it blocked",
        "user: my card got declined this morning",
        300,
    ],
    [
        "user_speech_during_agent",
        "on monday there was a payment of forty two dollars to city water",
        "honey can you grab the door",
        "user: can you read me my recent transactions",
        300,
    ],
    ["user_pause", "how can I help you today", "I wanted to ask about", "", 300],
    [
        "user_pause",
        "how can I help you today",
        "what's the routing number for my checking account",
        "",
        300,
    ],
    ["user_pause", "how can I help you today", "can you book me a table for two tonight", "", 300],
    [
        "user_pause",
        "can you confirm the payee for me",
        "this is ridiculous, I've asked three times, get me a real person",
        "user: my transfer never arrived\nagent: let me check that for you",
        300,
    ],
]

with gr.Blocks(title="floorcall") as demo:
    gr.Markdown(
        "# floorcall\n"
        "A millisecond decision layer for voice agents: **is the user done talking, was that "
        '"uh-huh" or a real interruption, what do they want, and do they need a human**, '
        "answered together in one pass of a fine-tuned, calibrated "
        "[Laya](https://huggingface.co/convaiinnovations/laya) encoder. "
        f"Model: [{CONFIG['model_id']}](https://huggingface.co/{CONFIG['model_id']}) · "
        "code: [github.com/yashraz23/floorcall](https://github.com/yashraz23/floorcall)"
    )
    with gr.Tab("Replay"):
        gr.HTML(space.replay_header(REPLAY))
        pick = gr.Dropdown(CHOICES, value=CHOICES[0][1], label="Scripted call", interactive=True)
        view = gr.HTML(show_script(CHOICES[0][1]))
        pick.change(show_script, pick, view)
    with gr.Tab("Try it"):
        gr.Markdown(
            "One live decision on this Space's CPU. The text is normalized as a speech recognizer "
            "would deliver it (lowercase, no punctuation) before the model reads it. The examples "
            "are written by hand."
        )
        with gr.Row():
            with gr.Column():
                event = gr.Radio(
                    [
                        ("the user paused (user_pause)", "user_pause"),
                        (
                            "the user spoke over the agent (user_speech_during_agent)",
                            "user_speech_during_agent",
                        ),
                    ],
                    value="user_speech_during_agent",
                    label="Event",
                )
                agent_last = gr.Textbox(label="The agent's last words", lines=2)
                user_words = gr.Textbox(label="What the user said", lines=2)
                history = gr.Textbox(
                    label="Earlier turns, one per line (user: … / agent: …)", lines=3
                )
                silence = gr.Slider(
                    300, 2000, value=300, step=100, label="Silence so far, ms (pause only)"
                )
                go = gr.Button("Decide", variant="primary")
            with gr.Column():
                out = gr.HTML()
        gr.Examples(EXAMPLES, [event, agent_last, user_words, history, silence])
        go.click(decide, [event, agent_last, user_words, history, silence], out)
    with gr.Tab("Results"):
        gr.Markdown((DATA / "results.md").read_text(encoding="utf-8"))
        for name, caption in CONFIG["figures"]:
            gr.Image(
                str(DATA / "figures" / name), label=caption, show_label=True, interactive=False
            )

if __name__ == "__main__":
    threading.Thread(target=processor, daemon=True).start()  # warm the model while the UI loads
    demo.launch()
