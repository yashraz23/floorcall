# CLAUDE.md: floorcall

> Originally drafted as "cadence". Renamed to floorcall on 2026-09-27 after the collision check:
> "cadence" is taken by cadence-workflow/cadence on GitHub and ai4bharat/Cadence on Hugging Face.
> Claude Code reads this file as standing context every session.
>
> **Read `docs/DECISIONS.md` before acting on this spec.** It records every place the build
> diverges from what is written here, with the reason. Where the two disagree, DECISIONS.md wins.

---

## 1. What this is

**floorcall is a millisecond decision layer for voice agents.** It sits inside a real-time voice pipeline and answers, in one forward pass of a small encoder model, the typed questions a voice agent must settle before it can act:

- Is the user done talking, or just pausing?
- The user spoke while the agent was talking. Was that "uh-huh", a real interruption, or noise?
- What does the user want, and is it something this agent handles at all?
- Should this conversation go to a human?

It runs **Laya** (open-weight, Apache-2.0, non-autoregressive "System 1" decision model, ~421M params), **fine-tuned and recalibrated** on real conversational data, plugged into a **Pipecat** voice pipeline.

**The deliverable is not the voice bot.** It is three things:
1. A fine-tuned, calibrated decision checkpoint published on Hugging Face with a real model card.
2. A results table comparing it against every sensible alternative on accuracy, calibration **and latency**.
3. A side-by-side demo where the same conversation runs through a naive agent and through floorcall, and the difference is audible.

---

## 2. Why a decision model, and not an LLM

This must be answerable in one sentence in any interview: **the latency budget makes an LLM call physically impossible.**

A voice agent's whole response loop (hear the user, decide, think, speak) has a budget of well under a second before the conversation feels broken. Speech recognition, the response LLM and speech synthesis already consume most of it. Every *inline* decision (whether to stop talking, whether to respond yet) must fit into the remaining tens of milliseconds. A generative LLM call cannot.

Laya answers several typed questions in **one encoder pass**, with calibrated probabilities, on a laptop GPU or CPU. That is not a cheaper option here. It is the only option.

**Design targets (ours, to be measured, not claims):**

| Component | Target |
|---|---|
| floorcall decision, GPU, batch 1 | p99 ≤ 50 ms |
| floorcall decision, CPU, batch 1 | p99 ≤ 100 ms |
| One pass with N questions vs N sequential single-question passes | measurably faster; report the ratio |

---

## 3. What already exists, and how we relate to it

Treat these as **baselines**, never as things to reimplement:

| Existing work | What it does | Our relationship |
|---|---|---|
| Pipecat Smart Turn v3 | Audio end-of-turn model, ~8 MB int8 ONNX | Mention in README; audio-only, not directly comparable on text |
| LiveKit text turn detector | End-of-turn from ASR transcript text | **Baseline for D1** if the text model is still obtainable (LiveKit has been moving to an audio detector; pin whatever version you use, and note its LiveKit Model License) |
| AssemblyAI / Deepgram Flux | End-of-turn inside the ASR | Out of scope; mention |
| Laya presets (`router_questions`, `guard_questions`, `triage_questions`) | Zero-shot generic decisions | **Baseline:** stock Laya |
| Jev (TypeSafe, hosted) | Hosted typed decisions | Optional baseline if API budget allows |

**Our angle is not end-of-turn alone.** It is the *combined* decision layer: turn-taking, barge-in classification, routing and escalation, answered together, fine-tuned on real conversational data, calibrated, and measured against a latency budget. No single component is novel; the measured combination is the contribution.

Never write "first" or "state of the art" anywhere.

---

## 4. The decision set

Decisions fire on **pipeline events**. Each event asks only the questions relevant to that moment, in **one pass**.

| Event | Fires when | Questions asked in one pass |
|---|---|---|
| `user_pause` | VAD detects silence while the user holds the floor | D1, D3, D4 |
| `user_speech_during_agent` | User audio detected while the agent's TTS is playing | D2, D4 |

### D1: `turn_complete` (noul)
*"Has the user finished their turn and is now waiting for a response?"*
- Data: Switchboard Dialog Act corpus (SwDA, on Hugging Face). Complete = a full utterance at a real turn end. Incomplete = the same utterance truncated at a random word boundary, plus utterances SwDA marks as continued across the other speaker (`+` tag).
- Baselines: fixed silence-timeout policy; LiveKit text turn detector.

### D2: `barge_in` (choice)
*"The user spoke while the agent was talking. What was it?"*

| Option | Criterion |
|---|---|
| `backchannel` | Listener feedback that does not claim the turn: "uh-huh", "right", "yeah", "is that right?" |
| `interruption` | The user wants the floor: a question, objection, correction or new request |
| `noise` | Non-speech, a fragment, laughter, or speech not directed at the agent |

- Data: SwDA tags `b` (backchannel, ~38k utterances) and `bh` (backchannel in question form) → `backchannel`. Substantive acts (statements, questions, directives) by the non-floor speaker → `interruption`. Non-verbal and abandoned utterances → `noise`.
- **Hard subset (report separately):** utterances whose surface token appears in both classes, e.g. "yeah", "right", "okay", "no". "Yeah." is a backchannel; "yeah but that's not what I asked" is an interruption. Aggregate accuracy will be dominated by trivial "uh-huh" cases, exactly like Anchor's code-block integrity result. The hard subset is where the real result lives.

### D3: `route` (choice)
*"What does the user want?"* over the demo agent's intents plus `out_of_scope`.
- Demo agent domain: **banking support**.
- Data: CLINC150, banking domain (15 intents) plus its `oos` (out-of-scope) class.
- Why it matters: out-of-scope detection is where production voice agents embarrass themselves.

### D4: `escalate` (noul)
*"Is the user asking for a human, or showing frustration the agent is not resolving?"*
- Training data: may include LLM-generated examples, each tagged `source=synthetic`.
- **Test data: real customer messages only, hand-labelled by Yash, minimum 300, frozen before any training run.** Suggested source: the public customer-support-on-Twitter corpus.

### D5 (stretch): `injection` (noul)
Spoken prompt-injection attempts. Baseline: `laya.guard_questions()`. Data: public prompt-injection datasets. Only if everything above ships.

---

## 5. State schema

Laya takes a **state** (text or JSON) plus typed questions. Every state **must fit in one model window**. Never rely on Laya's multi-window mode: its README states that across windows the returned probability is the deciding window's, not a calibrated number for the whole input.

```json
{
  "agent_speaking": true,
  "agent_last_utterance": "your card ending 4417 was declined because the",
  "user_partial": "yeah but why",
  "recent_turns": [
    {"speaker": "user", "text": "i tried to pay at the grocery store"},
    {"speaker": "agent", "text": "i can help with that let me check your card"}
  ]
}
```

**State packer rules** (`src/floorcall/state.py`):
- Hard token budget: stay under the checkpoint's window with margin. Measure with the actual tokenizer, never estimate.
- Priority when truncating: `user_partial` in full → `agent_last_utterance` (tail) → `recent_turns` newest first.
- Log every packed state's token count. A state that exceeds the budget is a bug, not a truncation.

**Critical normalization:** streaming ASR output usually has **no punctuation and no casing**. SwDA text has both. Train on punctuated text and the model learns "ends with a period → turn complete", which is leakage that will not exist at inference time.
- **All training, test, replay and live text goes through the same normalizer:** lowercase, strip punctuation, collapse whitespace.
- Unit test this. Then add an ablation row showing what happens without it, because it is a good story.

---

## 6. Policy layer

The model outputs probabilities. The **policy** turns them into actions. Keep it explicit and configurable in `config.py`, because every threshold is a point on a tradeoff curve.

```
on user_pause:
    if p(turn_complete) >= θ_yield:          respond
    elif silence_ms >= max_wait_ms:           respond anyway (safety net)
    else:                                     keep listening
    if p(escalate) >= θ_escalate:             hand off to human flow
    route := argmax(route), unless p(out_of_scope) >= θ_oos → polite out-of-scope reply

on user_speech_during_agent:
    if barge_in == interruption and p >= θ_interrupt:   stop TTS immediately, flush audio, listen
    elif barge_in == backchannel:                        keep talking
    else (noise):                                        ignore
```

**The two headline tradeoff curves** (this project's equivalent of Anchor's abstention curve):
1. **θ_interrupt sweep:** false stops (agent halts for "uh-huh") vs missed interruptions (agent talks over a real objection).
2. **θ_yield sweep:** premature responses (agent cuts the user off) vs added response delay.

Because the model is calibrated, a deployer picks a point on each curve deliberately instead of guessing a timeout.

---

## 7. Architecture

```
 browser mic ──► WebRTC transport ──► Silero VAD ──► streaming STT ──┐
                                                                     │ transcripts + VAD events
                                                                     ▼
                                                   ┌──────────────────────────────┐
                                                   │  FloorcallDecisionProcessor    │
                                                   │  • builds state (packer)     │
                                                   │  • one Laya pass per event   │
                                                   │  • applies policy            │
                                                   │  • emits decision events     │──► WebSocket ──► UI panel
                                                   └──────────────┬───────────────┘
                                         respond / stop TTS / route / escalate
                                                                  ▼
                                                     response LLM ──► TTS ──► WebRTC ──► speaker
```

- The decision processor is a **Pipecat FrameProcessor**. Laya runs **in-process** (no HTTP hop) on the latency path. Laya's own Jev-compatible HTTP server is only for the optional standalone service.
- Every decision event carries: event type, questions asked, answers with probabilities, action taken, and **measured latency in ms**.

---

## 8. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language / env | Python 3.11, `uv` | Same as Anchor |
| Decision model | `laya` (**pin exact version**) + PyTorch | The subject of the project |
| Voice framework | Pipecat (**pin exact version**) | Open, Python, vendor-neutral; FrameProcessor fits the design |
| Transport | Pipecat's local WebRTC transport | Runs in a browser on localhost, no account needed |
| VAD | Silero (ships with Pipecat) | Standard |
| STT | Deepgram streaming (default); faster-whisper local (fallback) | Quality for the demo; a no-key path for cloners |
| Response LLM | Groq or Anthropic (configurable) | Groq for low latency |
| TTS | Cartesia (default); Kokoro or Piper local (fallback) | Same reasoning |
| Web app | FastAPI + WebSocket | Decision events stream to the UI |
| Frontend | React + Vite | Yash built SAGE's React frontend |
| Eval | scikit-learn, own ECE/Brier implementation, matplotlib | Calibration metrics must be ours and tested |
| Tracking | Weights & Biases (offline mode is fine) | Same pattern as Anchor |
| Release | Hugging Face Hub (model + dataset cards) | Public, citable artifact |
| Quality | pytest, ruff, mypy, GitHub Actions | Non-negotiable |

**Environment warnings:**
- **Do not install TensorFlow in this environment.** Laya's README documents that `laya.load()` can deadlock when TF is present, because transformers probes for it at import.
- Laya's API is new and changing fast. **Wrap it in one adapter module** (`src/floorcall/model/laya_adapter.py`) so version changes touch one file. Read the installed version's docs; do not trust snippets, including this file's.
- Use the **English** checkpoint. The project is English-only; the multilingual router is out of scope.

---

## 9. Repo structure

```
floorcall/
├── CLAUDE.md
├── README.md                 # hero video, results tables, curves, quickstart
├── pyproject.toml
├── .env.example
├── .github/workflows/ci.yml
├── docs/DECISIONS.md         # log of every divergence from this spec, with reasons
├── src/floorcall/
│   ├── config.py             # every threshold and knob; no magic numbers elsewhere
│   ├── normalize.py          # ASR-style text normalization (shared everywhere)
│   ├── state.py              # state packer with hard token budget
│   ├── questions.py          # the decision set as typed question schemas
│   ├── model/
│   │   └── laya_adapter.py   # the ONLY file that imports laya
│   ├── policy.py             # probabilities → actions
│   ├── pipeline/
│   │   ├── processor.py      # FloorcallDecisionProcessor (Pipecat)
│   │   ├── bot.py            # full voice agent assembly
│   │   └── naive.py          # baseline agent: VAD timeout + stop-on-any-speech
│   ├── data/
│   │   ├── swda.py           # D1 + D2 construction
│   │   ├── clinc.py          # D3 construction
│   │   ├── escalate.py       # D4 construction
│   │   └── splits.py         # conversation-disjoint splitting
│   ├── train/                # fine-tune + temperature calibration
│   ├── evaluate/
│   │   ├── metrics.py        # accuracy, macro-F1, ECE, Brier, reliability bins
│   │   ├── baselines.py      # majority, stock laya, livekit, prompted LLM, (jev)
│   │   ├── latency.py        # p50/p95/p99 benchmark harness
│   │   └── curves.py         # θ sweeps
│   ├── replay/               # scripted conversations, no audio, no keys
│   └── web/                  # FastAPI + WebSocket server
├── web-ui/                   # React app
├── data/
│   ├── raw/                  # gitignored
│   ├── processed/            # built by scripts, gitignored except cards
│   └── test_frozen/          # frozen test sets + SHA256 manifest, committed
├── demo/scripts/             # replay conversation JSON files
└── tests/
```

---

## 10. Data rules (these protect every number in the README)

1. **Conversation-disjoint splits.** All SwDA utterances from one conversation land in the same split. Splitting by utterance leaks speaker and topic across train and test. `splits.py` must assert this, and a test must fail if it is violated.
2. **Freeze test sets first.** Build, hash (SHA256) and commit `data/test_frozen/` before the first training run. Any later change to a test set is a new versioned file, never an edit.
3. **Synthetic data never enters a test set.** LLM-generated examples are allowed in training for D4 only, and every row carries a `source` field.
4. **Same normalizer everywhere.** Train, test, replay and live all go through `normalize.py`.
5. **Report the hard subsets separately** (D2 ambiguous tokens; D1 truncations that end on a complete-looking clause).
6. **Record class balance** for every split in the README. With imbalanced classes raw accuracy is meaningless; always report macro-F1 alongside it.
7. **Check licenses** for SwDA, CLINC150 and any customer-support corpus before publishing a derived dataset. Some are non-commercial; that is fine for a portfolio release but must be stated on the dataset card.

---

## 11. Training

- Start from the official Laya fine-tuning notebook (built for Kaggle's free 2× T4). **Read its data format and reproduce it exactly** (states, questions, gold answers) rather than inventing one.
- Port it to run locally on the RTX 5070 Ti (12 GB): bf16, gradient checkpointing, small batch with gradient accumulation. Confirm this in milestone 0 before relying on it; the Kaggle notebook is the fallback.
- **One multi-task checkpoint** trained on all decisions together. (Stretch ablation: per-decision checkpoints vs multi-task.)
- **Calibration:** after training, fit temperatures on a held-out calibration split (never the test split), as the notebook does. Report ECE and Brier before and after.
- Every training run logs its config, data manifest hashes and git SHA.

---

## 12. Evaluation: the tables the README must contain

### Table A: quality per decision
Rows: majority class · stock Laya (zero-shot) · fine-tuned Laya · fine-tuned + temperature scaling · LiveKit text turn detector (D1 only) · prompted small LLM (e.g. Groq-hosted) · Jev (optional).
Columns: accuracy · macro-F1 · ECE · Brier · hard-subset accuracy.

### Table B: latency
Batch size 1 (real time is batch 1), 50 warmup iterations, ≥1,000 timed iterations, fixed input set.
Rows: floorcall GPU · floorcall CPU · floorcall one pass with 3 questions vs 3 sequential passes · LiveKit detector CPU · prompted LLM (end to end, network included).
Columns: p50 · p95 · p99 · fits budget (yes/no).

### Table C: robustness
Same test set, text degraded to look like real ASR output: word drops, substitutions, missing trailing words. Report accuracy per noise level.

### Table D: ablations
- Without ASR-style normalization (punctuation leakage).
- Without `recent_turns` in the state.
- Without `agent_last_utterance` in the state.

### Curves
- θ_interrupt: false stops vs missed interruptions.
- θ_yield: premature responses vs added delay.
- Reliability diagrams, before and after calibration, for D1 and D2.

---

## 13. Demo

**Replay mode (the reproducible one; must work with zero API keys):**
```bash
uv run floorcall replay demo/scripts/*.json --compare naive
```
Scripted conversations (turn text, timing, agent speaking windows) run through **both** the naive agent and floorcall, and the command prints where they diverge: "naive stopped for 'uh-huh' at 00:07; floorcall kept talking". This is what makes the repo demoable by anyone who clones it.

**Live mode:** browser at localhost, talk to the banking agent.

**The UI shows, in real time:**
- Live transcript with partial ASR.
- **Decision timeline:** each event, the questions asked, probability bars per answer, the action taken, and the latency in ms.
- **Latency waterfall per turn:** VAD, STT, decision, LLM first token, TTS first audio.
- A **naive / floorcall toggle**.

**README hero:** a 30 to 45 second screen recording with audio, split screen: the naive agent stops for "uh-huh" and cuts the user off; floorcall handles both correctly, with the decision panel visible. This one video carries the project.

**Hugging Face release:** the fine-tuned checkpoint with a model card containing Tables A to C, intended use, known failure modes and license notes.

---

## 14. Milestones

### Milestone 0: spike (one evening, before committing)
- Install laya; run `predict` on 20 hand-written voice states covering every decision.
- Measure batch-1 latency on the 5070 Ti and on CPU.
- Run one epoch of fine-tuning on a tiny subset locally to confirm it fits in 12 GB.
- **Exit:** latency numbers in hand, and local fine-tuning confirmed or a documented decision to use the Kaggle notebook.

### Milestone 1: data (week 1)
- SwDA → D1 and D2; CLINC150 banking + oos → D3; D4 hand-labelled test set.
- Conversation-disjoint splits, frozen hashed test sets, normalizer with tests, dataset cards.
- **Exit:** Table A's majority-class and stock-Laya rows filled.

### Milestone 2: train and evaluate (week 2)
- Fine-tune, calibrate, fill Tables A to D, produce curves and reliability diagrams.
- **Exit:** every number in the README reproducible from one command.

### Milestone 3: pipeline and replay (week 3)
- Pipecat bot, decision processor, naive baseline agent, replay mode.
- **Exit:** `floorcall replay --compare naive` produces the divergence report.

### Milestone 4: UI and release (week 4)
- React UI, WebSocket events, latency waterfall, recorded hero video, Hugging Face release, README.
- **Exit:** a stranger can clone, run replay mode with no keys, and understand the result from the README alone.

### Cut list, in order, if behind
1. D5 injection.
2. Jev baseline.
3. Per-decision vs multi-task ablation.
4. Local STT/TTS fallbacks (keep hosted defaults only).
5. Table C robustness.

**Never cut:** frozen conversation-disjoint test sets, calibration metrics, the latency table, replay mode, the hero video (a 30–45 s screen recording of the Space's Replay tab; DECISIONS.md D-045).

---

## 15. Instructions for Claude Code

- **Never fabricate a number.** Unmeasured cells are `TODO`, never a plausible value. Every number in the README must be reproducible from a committed command.
- **Never train, calibrate or tune thresholds on a test split.** If a change requires looking at test results to decide, stop and say so.
- **Ask before adding a dependency.** Especially anything that pulls in TensorFlow.
- **Only `laya_adapter.py` imports laya.** Verify API calls against the installed version.
- **Test first** for `normalize.py`, `state.py`, `splits.py`, `metrics.py` and `policy.py`. ECE and Brier need tests against hand-computed examples.
- **Do not silently change** thresholds, token budgets, split seeds or normalization. They are experiment variables: change them in `config.py` and note it in the commit.
- **Explain design decisions** in comments or commit bodies, especially in the policy and data construction. Yash must be able to defend every one in an interview.
- Prefer small, reviewable diffs. Conventional commits (`feat:`, `fix:`, `data:`, `eval:`).
- Secrets live in `.env`, never committed. Replay mode must never require a key.
- Log every choice that diverges from this spec in `docs/DECISIONS.md`, with the reason, so a later session does not undo it.

---

## 16. Known risks

| Risk | Mitigation |
|---|---|
| A text-only model can't hear prosody, so some turn ends are ambiguous in text | Policy combines p(turn_complete) with measured silence; state this limitation plainly in the README |
| SwDA is 1990s telephone conversation, not user-to-agent speech | Report it as a domain gap; D3/D4 use closer-domain data; the live demo shows transfer qualitatively |
| Laya's API changes mid-project | Pinned version plus the single adapter module |
| Pipecat plumbing eats the schedule | Replay mode is independent of the live pipeline; the core results never depend on audio working |
| D4 test set is small and hand-labelled | State n and show confidence intervals; don't over-claim |

---

## 17. What this project puts on the resume

Voice AI · real-time systems · Pipecat · WebRTC · decision models (Laya) · fine-tuning · probability calibration (temperature scaling, ECE, Brier) · latency engineering (p99 under a budget) · conversation-disjoint evaluation · Hugging Face model release · React
