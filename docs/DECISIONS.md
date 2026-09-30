# Decisions

Every place the build diverges from `CLAUDE.md`, or settles something the spec left open,
with the reason. Newest last. A later session must not undo an entry here without adding a new
entry that says why.

Format: **D-NNN · date · spec section** — what was decided, then why.

---

**D-001 · 2026-09-27 · header** — The project is called **floorcall**, not cadence.
"cadence" is taken by `cadence-workflow/cadence` on GitHub (Uber's workflow engine, ~9.4k stars)
and by `ai4bharat/Cadence` on Hugging Face, a punctuation-restoration model for speech text, which
sits in this project's own domain. "floorcall" had no GitHub repositories, no PyPI package and no
Hugging Face models on the day it was checked. "Holding the floor" is the conversation-analysis
term for having the turn; the "call" is the decision about it.

**D-002 · 2026-09-27 · §8** — The TensorFlow warning comes from Laya's **CI config**, not its
README. `laya-src/.github/workflows/ci.yml` at commit `9d95567` says: "transformers probes for
TensorFlow at import; when TF is present its abseil runtime can deadlock model construction."
The rule (never install TF here) stands. `tests/test_environment.py` fails if `tensorflow`
becomes importable.

**D-003 · 2026-09-27 · §1, §2, §12 Table B** — "One pass" means **one batched forward call**, not
one encoder pass shared across questions. In Laya 0.3.21 each question is its own input row:

```
[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
```

(`laya/common.py::build_sequence`). `Agent.predict` tokenizes the state once, builds one row per
question, and runs all rows through the encoder together. So asking three questions costs one
forward call with batch dimension 3, and the encoder reads the state three times. The saving over
three sequential calls is batching parallelism plus the per-call overhead paid once. That is
likely a real win on GPU and possibly a small one on CPU, where Laya's own README reports batching
may not help. Table B measures this directly instead of assuming it. README and model-card wording
must say "one forward call" or "one batched pass", never "one encoder pass".

**D-004 · 2026-09-27 · §5** — The state token budget is derived from the checkpoint, per event.
The English `laya` checkpoint has `max_len=512` and `head_max_len=192`. The instructions and
options take the front of each row, and the state gets whatever room remains
(`max_len - len(prefix) - 1`). That room differs per question, so the packer's budget for an event
is the **minimum room across the questions that event asks**, computed by the adapter with the real
tokenizer on the real question schemas, minus a safety margin set in `config.py`. Laya
right-truncates a dict state silently when it overflows. The adapter therefore re-checks every
packed state and raises if any row would truncate, which enforces the spec's "a state that exceeds
the budget is a bug".

**D-005 · 2026-09-27 · §4** — Every `noul` question (D1, D4, D5) carries explicit `true` / `false`
criteria. On the English checkpoint, a `noul` with no criteria renders a generic option pair that
"carries the whole decision, so it answers 'no' whatever the state" (Laya README, Honest limits,
issue #156). `choice` labels avoid boolean words (`yes` / `no` / `true` / `false`) for the same
reason.

**D-006 · 2026-09-27 · §4 D1/D2** — SwDA is read from the original corpus CSVs at
`https://github.com/cgpotts/swda/raw/master/swda.zip`, not through Hugging Face `datasets`. The HF
entry `cgpotts/swda` contains only a loading script, which current `datasets` releases no longer
execute, and it has no parquet conversion. The CSVs are what that script downloaded anyway.
Licence: CC BY-NC-SA 3.0, so the derived dataset is non-commercial and must say so on its card.

**D-007 · 2026-09-27 · §8** — The torch build is chosen explicitly, through two mutually exclusive
dependency groups: `cu130` (a default group, so a plain `uv sync` / `uv run` on the GPU box gets
it) and `cpu` (CI: `uv sync --no-default-groups --group dev --group cpu`). On Windows the PyPI
torch wheel is CPU-only, and the RTX 5070 Ti Laptop GPU (Blackwell, compute capability 12.0,
sm_120) needs CUDA 12.8 or newer. cu130 matches the installed driver (CUDA 13.1). Verified:
`torch 2.14.0+cu130` lists `sm_120` and runs a bf16 matmul on the card.

**D-008 · 2026-09-27 · §13** — The GitHub repository stays **private** until Table A has measured
rows, then goes public. The history is kept either way. (Yash's call.)

**D-009 · 2026-09-27 · §4 D4, §10.2** — All four test sets, including Yash's hand-labelled D4 set,
are frozen before the first training run, as the spec says. Training waits on the D4 labels.
(Yash's call; the alternative was a D1–D3 run first.)

**D-010 · 2026-09-27 · §12 Table B** — Laya silently falls back to CPU when it cannot place the
model on the GPU (`Agent.__init__`, and `_infer` on OOM). It prints a warning but does not raise.
On a Blackwell card with a torch build that lacks sm_120 support, a "GPU" latency row would then
quietly be a CPU number. The latency harness reads the device the model actually runs on after
loading and after timing, and refuses to write a GPU row if either reads anything but `cuda`.

**D-011 · 2026-09-27 · §8** — The torch variants are dependency groups, not extras. They were
first written as extras (`--extra cu130`). A later bare `uv run` then re-synced the environment to
the no-extra set and silently replaced `torch 2.14.0+cu130` with `2.14.0+cpu`. Any "GPU" run after
that would have been a CPU run without an error. uv has no environment variable for a sticky extra,
but it does have `default-groups`, so the CUDA build is now a default group and the CPU build is
opted into explicitly.

**D-012 · 2026-09-27 · §2, §12 Table B** — The GPU latency path will use a **CUDA-graphed
forward**. The M0 spike measured the eager path at p99 72–82 ms, against the 50 ms target. The
forward is launch-bound: a single 128-token row still took 39 ms, for a few milliseconds of real
compute. An exploratory capture of the Laya forward as a `torch.cuda.CUDAGraph` produced
bit-identical logits and brought `user_pause` to p99 ≈ 31 ms. It needs no new dependency. Laya's
own fast path (`laya[fast]`, TileLang + CUDA graphs) would need `tilelang`, and was not tried. In
M2 the adapter gains a graphed forward over fixed shape buckets, with a parity test against eager.
Table B reports both eager and graphed rows. Table A is computed on logits from the forward that is
actually served, and the parity test is what makes that claim checkable. (docs/spike-m0.md §3)

**D-013 · 2026-09-27 · §11** — Training keeps the English checkpoint's own geometry,
`max_len=512` and `head_max_len=192`. The notebook sets 1024 / 256, but that is the geometry of
the `laya-typed-decisions` checkpoint it produces. Serving at 512, the English default and the
length the packer budgets for, means training at 512 too. Also: bf16 autocast without a
GradScaler (the notebook's fp16 + scaler exists because T4s lack bf16), and one GPU with
accumulation 8, which keeps the notebook's 64-row effective batch. The M0 spike measured peak
memory at 8.4 GB on worst-case 512-token rows, so training runs locally and the Kaggle notebook
is not used. (docs/spike-m0.md §4)

**D-014 · 2026-09-27 · §2, §12 Table B** — The CPU target (p99 ≤ 100 ms) is kept as written, and
the M0 spike says the stock path misses it by about 10x: `user_pause` at p50 0.8 s sequential and
1.2 s batched, at roughly 490 GFLOPs per call. Thread tuning will not close a 10x gap. Table B
reports the measured CPU numbers with "fits budget: no" rather than moving the target. INT8 ONNX
(`laya[onnx]`, which adds `onnxruntime`) is the plausible mitigation. It needs Yash's approval
for the dependency and sits on the cut list below Table C.

**D-015 · 2026-09-27 · §4 D1/D2** — How SwDA becomes labels. The caller being decided about plays
the user, the other caller plays the agent.
*D1 turn_complete*: `complete` means the segment closes its slash unit (`/`, not abandoned `-/`)
and the other caller then takes the floor. `continuation` (false) means the segment does not close
its unit, the other caller only backchannels, and the speaker resumes with a `+` continuation.
`truncation` (false) is a complete turn cut at a random word boundary inside its last utterance.
Excluded as unlabelable from text: a closed unit, then only a backchannel, then a new unit from the
same speaker; and a turn cut off by the other caller after an unfinished or abandoned segment.
*D2 barge_in*: see D-016.
*Who holds the floor*: backchannels (`b`, `bh`), noise (`x`, `%`), and **minimal responses** do not
take or hold it. A minimal response is `aa`, `bk`, `ba`, `na`, `ny` or `b^m` of at most 3 words
(`DataSettings.minimal_response_max_words`). SwDA's coders tag a short "yeah" by function, and
81-100% of those tags are three words or fewer. The first build counted them as turns, which
flipped the floor: a speaker simply going on after the listener's "Yeah." (`aa`) looked like a
barge-in, and a listener's "Yeah." made the speaker's unit "complete". Fixing it removed about 12k
D1 "complete" labels that were really the ambiguous excluded case. A residual 0.6% of D2 test rows
have an agent turn such as "No," tagged as a real answer (`nn`), which does take the floor.
*Text*: SwDA markup is stripped to spoken words (`clean_text`). Fillers ("uh", "um") are kept,
because ASR can emit them. The live pipeline must be configured to keep them too (Deepgram
`filler_words=true`), or train and live text will differ.

**D-016 · 2026-09-27 · §4 D2** — D2 labels a caller's **whole run** of utterances, not one row,
and the text is its first 12 words. SwDA segments by dialog act, so "Yeah, but that's not what I
asked" is stored as two rows, "Yeah," (`b`) and "but that's ..." (`sd`). Measured on single rows,
only 11 of 14,376 train interruptions start with "yeah", so the spec's central hard case barely
exists. A recogniser delivers the merged string. So the run is an interruption if any utterance in
it claims the floor, a backchannel if every utterance is one, noise if all of it is noise, and
excluded otherwise. After the change, 4,067 train interruptions open with "yeah". The 12-word cap
(`d2_partial_words`) approximates the partial transcript at decision time.
*Hard subsets* are defined from train statistics only. D2: a backchannel form qualifies when at
least 20 train backchannels are exactly it, at least 20 train interruptions open with it, and each
class is at least 20% of its rows. Without the share floor, "uh huh" qualified on volume alone (8%
interruptions), and "hard" became 60% of the test set; with it, hard is 38%, built from yeah, right,
oh, okay, yes, sure, really and similar. D1: a truncation is hard when its last word w has
(train complete turns ending in w) / (train truncations ending in w) >= 1, i.e. a last-word
heuristic would call it complete.

**D-017 · 2026-09-27 · §4 D3** — D3 is built from `clinc/oos-eval` `data/data_oos_plus.json`
(counts identical to HF `clinc/clinc_oos` "plus"), with intent names inline. CLINC's own splits
are used: train -> train, val -> calib, test -> test. Calib and test rows whose normalized text
duplicates a train row are dropped (one test row). Only CLINC's `oos` queries are out_of_scope;
the other nine domains' queries are unused, so the label definition stays CLINC's. Each query is
packed as a `user_pause` state with no history, and the agent's last line is one of five fixed
openers, chosen by a hash of the row id. Test balance: 30 per intent, 1,000 out_of_scope, so
majority-class accuracy is 0.69. Macro-F1 is the headline D3 number.

**D-018 · 2026-09-27 · §10.2** — The first D1/D2 test files were frozen, then **regenerated before
they were ever committed or evaluated**, when review found the minimal-response bug (D-015). The
freeze guard refused the rebuild, as designed. The uncommitted files were deleted and v1 was
rebuilt. The CLINC set came back with an identical hash (`9d1850a8…`), which also shows the build
is deterministic. From the first commit of `data/test_frozen/`, any change is a new version.

**D-019 · 2026-09-27 · §4 D4** — The D4 labelling pool. Source: "Customer Support on Twitter"
(Thought Vector, Kaggle, CC BY-NC-SA 4.0), `twcs.csv` via an unmodified Hugging Face mirror, pinned
by the file's content hash. Choices:
*Banking threads only*: the eight banking and payments brands in the corpus (BofA, Amex, PayPal,
Chase, Wells Fargo, Citi, Visa, KeyBank), so the D4 number describes the demo agent's domain.
*Split by thread*, with the same hash scheme as SwDA conversations (root tweet id), so test, calib
and any future training rows never share a thread. *One message per thread.*
*Four equal strata*: escalation-word cue (`ESCALATION_CUES`) x whether the agent had already
replied. The cue axis exists because 300 natural-rate labels might hold too few positives. The
context axis exists because 74% of messages open a thread, while "frustration the agent is not
resolving" needs a prior agent reply. As a consequence the test set's positive rate is not the
natural rate. Every D4 number must say so, and results are also reported per stratum. The
labelling tool never shows the stratum. "Hard" for D4 means no cue word, so a keyword match cannot
find it.
*Pool*: 400 test and 200 calib candidates, committed as `data/labels/escalate.candidates.v1.jsonl`
before any label exists. The calib labels are what theta_escalate and the D4 temperature are fitted
on, since neither may come from test. Freezing requires at least 300 true/false test labels; skips
never enter a set. Guidelines: `docs/labelling-escalate.md` v1, recorded on every label.
*Cleaning*: agent signatures (`^MG`, `^Clarissa`) are stripped from agent tweets only, since a
trailing `*word` on a customer tweet may be content. A simple function-word test keeps non-English
messages out of the pool; the labeller can skip anything it misses.

**D-020 · 2026-09-27 · §12 Table A** — How Table A is produced and read.
*README numbers are generated*: `floorcall eval readme` renders the tables from `results/table_a/*.json`
(each written by a committed command, with the test file's hash, code commit, checkpoint revision,
device and temperature inside). A cell with no file renders as TODO, and `tests/test_readme.py`
fails if README.md differs from the render, so no number can be typed in by hand.
*Majority baseline* predicts the train split's majority class with the **train class prior** as its
probabilities. The majority is taken from train because taking it from test is peeking. The prior,
rather than a one-hot, gives a meaningful Brier score and ECE.
*D1's hard subset is single-class*: truncations, all labelled incomplete. A constant "incomplete"
predictor, which is what the majority baseline is on D1, scores 1.000 there. The column is only
informative next to a model's overall accuracy, and the README says so under the table.
*Stock Laya* is scored from `logits_batch` with the checkpoint's own shipped temperatures, including
the clamped `choice:11+` bucket that sharpens D3 (docs/spike-m0.md §2). Every state is packed with
its event's budget, exactly as in serving.

**D-021 · 2026-09-28 · §11** — floorcall fits **one temperature per decision**, on the calib split,
and applies it itself to the raw logits (`evaluate/calibration.py`, `decider.py`). Laya keeps one
temperature per question type and option count (`noul:2`, `choice:3-5`, ...), and its notebook fits
one per type. D1 turn_complete and D4 escalate are both two-option `noul` questions, so under
Laya's scheme they would share a temperature fitted on a mixture of two unrelated tasks. The fit
minimises NLL over beta = 1/T, which is convex, with a golden-section search bounded to
[0.25, 10] (`EvalSettings`); a fit that ends on a bound is flagged. The temperatures live in
`floorcall_calibration.json` beside the checkpoint, with the calib files' hashes. A checkpoint
without that file (the stock one) uses Laya's shipped temperatures, so it behaves as Laya serves
it. The serving path therefore bypasses Laya's decode: `LayaDecider.logits_one` returns raw scores
from one batched call, and `Decider` packs, scores and applies temperatures.

**D-022 · 2026-09-28 · §12 Table B** — How the CUDA-graphed forward (D-012) is built and checked.
Graphs are keyed by (rows, sequence length rounded up to 32 tokens, options), captured on first
use, and share one memory pool; states in the same bucket share a graph. Two facts from the
installed transformers 5.17 source make padding safe. ModernBERT skips its attention mask when a
batch has no padding, a data-dependent branch, but `is_tracing()` is true during stream capture,
so a graph always takes the masked path. And end padding moves no RoPE position of a real token.
Measured against eager on frozen-test states: replay is bit-identical, max |dp| is 0.031, and
median forward time falls from 80.8 ms to 35.2 ms on full-length states. Argmax flipped on 1 of 240
and 4 of 160 decisions, and every flip had an eager top-2 margin of exactly 2^-8, one bf16 step:
exact ties, since the head scores under bf16 autocast. `tests/test_cuda_graphs.py` therefore
forbids flips with a real margin (> 0.02) rather than demanding a fixed agreement rate.

**D-023 · 2026-09-28 · §11** — Training data, mixture and guard. One multi-task checkpoint, trained
on items packed exactly as served: the same event budgets and the same StateSettings, via
`EvalSet.packed_states`. Each epoch draws fixed quotas per task, because the train sets differ 30x
(D1 49k rows, D3 1.75k): 20,000 D1, 20,000 D2, 5,250 D3 (1,750 repeated 3x) and 5,000 D4, the last
provisional until D4's training source is decided. Tasks over quota are subsampled without
replacement each epoch. Targets are one-hot, because every label source here is hard; calibration
comes from the calib-split temperature (D-021), not from smoothing. The loop reproduces the
notebook: the RLCD policy gradient plus soft CE, bf16 autocast, gradient checkpointing, 64-row
updates, cosine decay, and sigma annealed per epoch. `floorcall train run` **refuses to start**
unless all four test sets are frozen and match the manifest (`assert_test_sets_frozen`), which
enforces D-009 in code. The checkpoint is saved in Laya's layout with neutral temperatures (D-021),
then reloaded from disk, and its calib logits are what the temperatures are fitted on. The code was
tested without training anything real: the loop on a tiny stand-in model with the same forward
signature, and save/load as an untrained round trip of the stock weights through plain
`laya.Agent`, which is logit-exact on CPU.

**D-024 · 2026-09-28 · §12 Table C** — The ASR noise model. The spec names the kinds of noise (word
drops, substitutions, missing trailing words) but not a model. In normalized text, each user word
is deleted with probability level/2 or replaced with probability level/2 by a word drawn uniformly
from the 5,000 most common train words. Then, with probability level, the last one or two words
are dropped, as when a partial transcript lags the speaker. Levels: 0, 0.05, 0.1, 0.2. Only user
words are degraded (`user_partial` and user turns in the history); agent lines come from our own
TTS input and are known exactly. The noise is seeded per (row id, level), so every model sees
identical degraded text. The level-0 row must equal Table A's calibrated row, which is a built-in
consistency check.

**D-025 · 2026-09-28 · §12 Table D** — Ablations are **separately trained checkpoints**, not the
main model with a field blanked at test time. A model trained with a field and tested without it
measures how much it relies on the field. Whether the field is worth having needs a model trained
without it. Each ablation is a named set of StateSettings overrides (`config.ABLATIONS`), applied
to training, calibration and evaluation and recorded in run.json. The evaluator reads the flags
back and refuses to score a checkpoint under different ones. The normalization ablation is scored
twice: on written text, where punctuation leaks the answer, and on ASR-style text, the live
condition. Cost: three extra training runs of about three hours each.

**D-026 · 2026-09-28 · §5, §12 Table B** — The packer's search got faster, and its output stayed
identical. The first GPU Table B run (f4c7e87) showed packing at 3.6–10 ms p50, and up to 19 ms p99,
of pure CPU time, because every candidate re-tokenized the whole JSON state: about 18 tokenizer
calls per state. Three changes: try "everything fits" first; a binary search over history turns in
place of a turn-by-turn scan; and memoized counts per exact string. Measured over all 29,807 frozen
test rows with the real tokenizer: about 18 → 6 calls per state, and about 5.2 → 2.7 ms per D1/D2
state. **0 of 29,807 packed states differ** from the previous packer, which is kept frozen as
`tests/reference_packer.py` and checked in `tests/test_packer_equivalence.py`.
Getting to 0 surfaced a real property of BPE. The first version differed on 1 row: a 299-word
monologue whose tail was kept at 288 words instead of 290 (329 vs 331 tokens, both within the
331-token budget). **The token count is not monotone in the number of tail words.** The first kept
word follows a quote with no leading space, and dropping the word before it can split that word into
more tokens, so the count wobbles by a token or two near the boundary. Two search orders can then
settle on different valid answers. The word-tail search therefore keeps the old probe sequence
exactly: after the fast path it searches [0, n], not [0, n − 1], and `fits(n)` is memoized, so it
makes the same decisions. History turns need no such care. Each turn adds at least ~10 tokens of
JSON wrapper against a 1–2 token boundary effect, so the count is strictly monotone in turns, and a
binary search equals the old linear scan.

**D-027 · 2026-09-28 · §6, §12 Curves** — How the two headline tradeoffs are drawn, and how their
operating points are chosen. *theta_interrupt*: a false stop is any non-interruption (backchannel or
noise) that stops the agent, also reported for backchannels alone; a miss is an interruption the
agent talks over. *theta_yield*: a premature response is an unfinished turn the agent answers. Text
has no clock, so "added delay" needs a model, and it is the policy's own. The pause event fires
after `vad_pause_ms` (300) of silence; a finished turn the model declines to answer waits for the
safety net at `max_wait_ms` (2,000). Added delay = fallback rate x 1,700 ms. **Operating points are
chosen on calib**: the smallest theta whose calib rate of the costly error is at or under its
target (`target_false_stop_rate`, `target_premature_rate`, both 5%), which keeps as many true stops
and prompt responses as the target allows. Test then reports what that theta does, and never chooses
it. A target calib cannot meet is reported as such, not quietly relaxed. Figures follow the dataviz
method: the reference palette's slots 1–2, assigned by entity (stock Laya always slot 1,
fine-tuned always slot 2) and validated with its script in both modes (CVD ΔE 24.7 light and 26.8
dark against a target of 8; contrast at least 3:1). One axis per chart, rendered light and dark and
served through `<picture>`.

**D-028 · 2026-09-28 · §12 Table B** — Table B rows are single runs, and the README says so. Two
runs of the GPU rows, at f4c7e87 and then at ae690c4 with the faster packer, differed at p99 by
10–20% in both directions: graphed barge-in 89.3 → 70.6 ms, but graphed pause 85.7 → 95.2 ms. The
packer change cut pause packing p50 from 3.8 to 1.3 ms, so the pause regression is not the code.
It is the machine: a 90 W-capped laptop GPU, thermal state, the Windows power plan, and background
processes. Packing and graph replay are deterministic; the tail is not. Before release, Table B is
rerun as repeated runs on an idle machine, and each cell reports the median of the runs' p99 with
their range. `floorcall eval latency` keeps its single-run shape until then, so every committed
number is exactly what one run of a committed command produced.

**D-029 · 2026-09-28 · §3, §12 Tables A and B** — **LiveKit's text turn detector is not a
baseline.** (Yash's decision.) The model is still published (`livekit/turn-detector`, main at
`fba34c38`, a small Llama-architecture causal LM that would have run on the transformers and torch
already installed). Its licence, the LiveKit Model License, §3.b, requires "not to use any LiveKit
Models on a standalone basis or with any frameworks other than LiveKit Agents", and "not to use any
LiveKit Materials or any output from, or results of using, LiveKit Models ... to improve or
otherwise develop any other models that are not LiveKit Models". Loading it in floorcall's harness
is standalone use. Running it through the LiveKit Agents plugin would add that framework as a
dependency and would still leave a published head-to-head in a grey area under the second clause.
Only the model's Hub metadata and its licence file were read. The model was never downloaded or
run, and nothing from it reached any data or result. The README says why the row is missing, and the row is removed from Table B.


**D-030 · 2026-09-28 · §4 D4, §12 Tables A and B** — Groq for D4 training labels and for the
prompted-LLM baseline. (Yash's decisions: LLM-label real messages, never generate synthetic ones;
the baseline on Groq; a $5 cap on all Groq use.)
*D4 training rows* are real customer messages from **train-split** banking threads (the same
thread-level hash as test and calib), one per thread, 6,000 sampled with a fixed seed. Any message
whose normalized text equals a test or calib message is also dropped, because a complaint can be
pasted into several threads. Each is labelled by `openai/gpt-oss-120b` (strict JSON schema, low
reasoning effort, temperature 0, fixed seed) from Yash's guidelines, whose seven decision rules
appear in the prompt word for word (checked by a test). "unsure" drops the message, as his skip
did. Rows carry `source=llm_labelled`; test sets carry no such rows (`tests/test_frozen_data.py`).
Every label, with its model and prompt version (`llm-labeller-v1`), is committed in
`data/labels/escalate.llm_labels.v1.jsonl`, so the train set rebuilds without the API.
*Agreement with Yash* (Cohen's kappa, accuracy, confusion, per stratum) is reported on calib and
test, and is **measurement only**. The prompt was fixed before it saw a test message, and nothing
is tuned or filtered on the test agreement.
*The baseline* is `openai/gpt-oss-20b`, a small model (21B total, about 3.6B active parameters) with
strict JSON schemas. It gets exactly Laya's inputs: the same packed state and the same question
wording. **Groq returns no log-probabilities** ("not yet supported by any of our models"), so the
baseline *states* a probability per option, and its ECE and Brier score measure stated
confidence. The README says so. D1 and D2 are scored on fixed, evenly spaced 1,000-row subsamples to
stay inside the cap, and D3 and D4 whole. The n is in each results file. For Table B, all three
user_pause questions go in one prompt, under the same protocol as every other row (50 warmup, 1,000
timed calls, cache bypassed), timed end to end with the network.
*The cap is enforced in code* (`floorcall.llm.groq.Ledger`). Every billed call's cost comes from the
usage the API returns, at Groq's published September 2026 rates (gpt-oss-120b $0.15/$0.60,
gpt-oss-20b $0.075/$0.30 per million tokens). Before a call, its worst case is reserved, and the call
is refused if spent + reserved + worst case would pass 95% of $5. A model with no known price is
refused. Responses are cached, so reruns cost nothing. The API key is a `SecretStr`, which keeps it
out of the settings dump that every run.json stores (a test asserts this).

**D-031 · 2026-09-28 · supersedes the provider parts of D-030** — **LLM calls go through
OpenRouter, not Groq.** (Yash's decision: Groq's paid tier is unavailable.) No Groq call was ever
made. Everything else in D-030 stands: the same models, by their exact ids and never OpenRouter's
`:batch` variants (`openai/gpt-oss-120b` labels, `openai/gpt-oss-20b` is the baseline); the same
prompts; the same $5 cap with its stop at $4.75.
*Cost is OpenRouter's own.* Every response carries `usage.cost` (OpenRouter: "the base currency is
US dollars"), and the ledger charges exactly that per call. A response without it would be charged
at its upper bound and marked `upper_bound`, so spend can only be over-counted, never under.
*The worst case is a bound, not a guess.* Every request sends `provider.max_price` at the rates
the budget was planned at (gpt-oss-120b ≤ $0.15/$0.60, gpt-oss-20b ≤ $0.075/$0.30 per million
tokens), so OpenRouter cannot route to a dearer provider. On 2026-09-28 prices across providers
ranged from $0.03/$0.17 to $0.35/$0.75 for gpt-oss-120b. Before each call, its worst case at that
ceiling is reserved against the stop.
*Only compliant providers.* `provider.require_parameters: true` routes only to providers that
support strict JSON-schema output, the seed and reasoning effort; providers without seed support
drop out. Reasoning is requested as `{"effort": "low", "exclude": true}`. Excluded reasoning
tokens are still billed and count against `max_tokens`, which rises to 800 to leave room for them.
*Which provider answered is recorded* per call in the ledger and per cached answer. Different
providers serve different quantizations (fp4, fp8, bf16), so labels can come from more than one
build of the same model; the record makes that inspectable.
*Stated probabilities, restated.* Behind OpenRouter some providers do return log-probabilities.
The baseline still states its probabilities, because support varies by provider and the answer is
a JSON object rather than one option token. The README says what its ECE measures.
The key is `OPENROUTER_API_KEY`, a `SecretStr`: it goes into one request header and nowhere else,
and a test asserts it stays out of the settings dump every run.json stores.

**D-032 · 2026-09-28 · refines D-030 and D-031 for the D4 labeller** — **The labeller runs on one
pinned full-precision endpoint, with prompt v2, and must pass a calib gate before it labels test
or train.** (Yash's decisions, option B after the v1 calib result.)
*Why.* On calib, `llm-labeller-v1` agreed with Yash's hand labels at accuracy 0.728, Cohen's kappa
0.434, escalate precision 0.506 and recall 0.833 (n = 195; `results/d4_labeller/llm-labeller-v1/`).
It escalated delay and inconvenience he labelled "no" (44 messages) and missed threats to leave or
close the account (9 messages). OpenRouter had spread the 200 calls over 8 providers serving 4-,
8- and 16-bit builds of the model, so the provider mix was a second source of variation.
*The pin.* `openai/gpt-oss-120b` is sent only to **Crusoe's bf16 endpoint** (`crusoe/bf16`): the
request's `provider.only` and `provider.order` hold just that slug, `allow_fallbacks` is false,
and `quantizations` is `["bf16"]`. A full slug is used because a bare provider slug matches every
endpoint that provider runs. The client also checks each response: one that names any provider
other than Crusoe is rejected, charged (it was billed) and not cached. Candidates on 2026-09-28,
the bf16 endpoints under the $0.15/$0.60 ceiling that support strict schemas, the seed and
reasoning: DekaLLM ($0.030/$0.180 per million tokens, 99.2% uptime over the last day), AkashML
($0.033/$0.187, 99.94%), Crusoe ($0.05/$0.25, 99.99%) and DeepInfra's bf16 endpoint (marked
degraded, 95.8%). Crusoe had the best uptime; at this volume the price difference is under a
cent. Cerebras's fp16 endpoint is above the ceiling. "Full precision" means the highest precision
served: gpt-oss is released with its mixture-of-experts weights in MXFP4, so a bf16 endpoint
serves those weights up-cast, and the fp4 and fp8 builds are excluded. The prompted-LLM baseline
(`gpt-oss-20b`) is not pinned; D-031 still applies to it.
*Prompt v2* (`llm-labeller-v2`) is v1 plus exactly two sentences, after the rules, one per error
direction: "Threats to leave or close the account are escalations even with no agent reply. Delay
or inconvenience alone is not, unless the customer says support has failed them." Yash's rules
are still quoted word for word (the existing test), and a test pins the two added sentences.
*The gate.* A prompt is judged on calib only. It is accepted if kappa ≥ 0.60 **and** escalate
precision ≥ 0.70 against Yash's labels (`labeller_min_kappa`, `labeller_min_escalate_precision`).
`data escalate-llm-label` refuses test and train unless calib results for that exact model,
prompt version and pinned endpoint pass the gate. If v2 misses, Yash sees the confusion table and
the remaining disagreements before anything else happens. At most two prompt revisions (v2, v3)
are allowed in total, all judged on calib. The hand labels are never edited. The test set is
labelled once, with the accepted prompt, and its agreement is measurement only.
*Files.* Labels are kept per prompt version, so v1's are never overwritten:
`data/labels/escalate.llm_labels.v1.jsonl` (v1) and `escalate.llm_labels.v2.jsonl` (v2). Each v2
label records the provider that served it. Results go to
`results/d4_labeller/<prompt version>/<split>.json`, with the pin, the serving providers, the gate
verdict (calib) and every disagreement.

**D-033 · 2026-09-28 · §4 D4, §10 rules 2 and 5** — **Guideline v2 for D4, and the D4 eval sets
become a blind relabel of a seeded sample under it.** (Yash's decisions, option A after D-032,
relabelling a subset to save labelling time.)
*Why the guideline changed.* The labeller's agreement with the v1 hand labels stalled at kappa 0.43
(prompt v1) and 0.42 (prompt v2) on calib (D-032). The calib disagreements showed the gap was
largely between guideline v1 as written and as applied. Several messages the written rule 2 covers
("No one could help me", "should I just take my business somewhere else?") were labelled n, and
some annoyance the written "n" rule covers was labelled y. Guideline v2 is Yash's rewrite: three
literal triggers, an explicit "n" list, and "if none clearly fits, n" in place of a skip. It is in
`docs/labelling-escalate.md` verbatim, and a test checks the doc and the code
(`labelling.GUIDELINE_V2`) against his text. v1's text stays in git (`808f0ef`).
*Informed by calib only.* The disagreements that prompted it were calib's. The LLM labeller has
never labelled a test message, so no test disagreement exists. The only D4 test-set results
produced so far are stock Laya's aggregates on test v1 (the Table A row with its 2x2 confusion,
and its reliability panel). None is per message, and the D4 majority row was never run.
*The sample.* Seed **20260928** (`DataSettings.d4_relabel_seed`). **200 from test, 100 from calib**
(`d4_relabel_sizes`), drawn from the v1 eval sets: 396 test and 195 calib messages. The 9 v1 skips
were never in an eval set. Each split's sample is allocated to the four sampling strata in
proportion to their sizes (largest remainder) and drawn uniformly within each stratum:
- test: cue/reply 51 of 100, cue/opener 49 of 98, plain/reply 51 of 100, plain/opener 49 of 98;
- calib: 26 of 50, 25 of 49, 25 of 49, 24 of 47, in the same order.
Only ids, splits, strata and whether a v1 label exists were read; never a label's value, an LLM
label or a disagreement. A test shows that flipping every v1 label leaves the sample unchanged.
The sample and its display order are committed in `data/labels/escalate.relabel_v2.sample.json`,
which is write-once.
*Blind relabel.* `floorcall label escalate-relabel` shows the 300 messages in a seeded shuffle, test
and calib interleaved. It shows the same context as the first pass (company, up to two earlier
turns, the agent's last reply, the message), with guideline v2 printed above every message. It
never reads the v1 labels or any LLM label, and shows no split or stratum. The keys are y and n
only. Labels go to `data/labels/escalate.labels.v2.jsonl` (`"guidelines": "v2"`); the v1 file
`escalate.labels.jsonl` is never written again.
*Eval sets v2.* Only the sampled messages stay in the D4 eval sets, with their v2 labels. The rest
leave, and v1 and v2 labels are never mixed in one set. Test v2 (200) and calib v2 (100) are frozen
as new versioned files with new hashes; the frozen v1 test file is not edited. **Test v2's 200 is
below the 300 minimum of CLAUDE.md §4**, by Yash's decision. Every D4 number is reported with its n,
and the majority-class and stock-Laya rows are re-run on test v2 with bootstrap 95% CIs.
Yash's self-agreement (v1 against v2 labels on the 300 sampled messages, kappa and confusion) is
reported as a measurement of the guideline change and of label consistency.
*Order.* Test is relabelled blind **before any fine-tuned model is evaluated**: no training run
has happened, and D4 training data does not exist yet.
*Labeller prompt v3* is guideline v2 verbatim, on the same pinned endpoint (`crusoe/bf16`). It is
judged on calib v2 against the same bars (kappa ≥ 0.60 and escalate precision ≥ 0.70). It is the
last prompt revision (D-032 allowed two: v2 and v3). Yash sees the result before test or train is
labelled.
*Outcome (2026-09-29).* Yash relabelled the 300 in one blind session, 01:14–01:55 UTC: 136
escalate, 164 not.
- **Self-agreement, v1 labels against v2 labels on the same 300 messages:** accuracy 0.783,
  **kappa 0.551**. By split: test 0.780 (kappa 0.541), calib 0.790 (kappa 0.570). Confusion (v1 →
  v2): y→y 82, y→n 11, n→y 54, n→n 153. Guideline v2 moves labels mostly one way, toward
  escalate (93 → 136).
- **Frozen.** Test v2, `escalate.test.v2.jsonl.gz`, sha256 `91994160…` (200 rows: 89 escalate,
  111 not). Calib v2, `escalate.calib.jsonl.gz`, sha256 `e9dbf51d…` (100 rows: 47/53). Both
  rebuild byte-identically from committed files (`tests/test_relabel.py`). The frozen test v1 is
  unchanged and still rebuilds to `e69e4823…`. D4 is evaluated on v2 (`build.TEST_VERSIONS`), and
  D1–D3 stay on v1.
- **Baselines on test v2**, with percentile-bootstrap 95% intervals (10,000 resamples of the rows,
  seed 20260927, `EvalSettings.bootstrap_*`):
  - majority (calib v2 prior): accuracy 0.555 [0.485, 0.625], macro-F1 0.357 [0.327, 0.385],
    ECE 0.025 [0.000, 0.095], Brier 0.495 [0.487, 0.504];
  - stock Laya: accuracy 0.665 [0.600, 0.730], macro-F1 0.625 [0.554, 0.693], ECE 0.092
    [0.057, 0.159], Brier 0.422 [0.396, 0.447].
  The D4 majority prior always comes from calib: D4's train rows will be LLM-labelled, so calib is
  its only hand-labelled split besides test.
- **Prompt v3 on calib v2: not accepted.** Accuracy 0.720, **kappa 0.420** (bar 0.60), escalate
  precision 0.952 (bar 0.70, met), recall 0.426. Confusion (Yash → LLM): y→y 20, y→n 27, n→y 1,
  n→n 52. All 100 calls were served by Crusoe bf16. v1 and v2 over-escalated; v3, reading the
  literal guideline, under-escalates. This was the last prompt revision. Test and train were not
  labelled with it, and the code refuses to until a prompt passes the gate.

**D-034 · 2026-09-29 · §4 D4, §11 · declared before any fine-tuned result exists** — **The
primary D4 model is trained on Yash's hand labels only. An LLM-labelled arm runs as an ablation.**
(Yash's decisions, options A and C after D-033.)
*The gate was likely unreachable.* D-032's bar, kappa ≥ 0.60 against Yash's labels, sat above his
own cross-guideline self-agreement: kappa 0.551, his v1 labels against his v2 labels on the same
300 messages (D-033). No more prompt revisions are made; `llm-labeller-v3` is final. Calib v2 is no
longer used for prompt selection. From here it serves only what calib is for: fitting D4's
temperature and choosing θ_escalate.
*The primary D4 model* is trained on Yash's hand labels only. This checkpoint is behind Table A's
"fine-tuned" and "fine-tuned + temperature" D4 rows, and it is the one released. Its D4 data is a
seeded random sample of 1,000 train-pool messages, labelled blind under guideline v2 with the
same tool and context as the test and calib relabel (`floorcall label escalate-train`). This is
declared on 2026-09-29, before any training run; no fine-tuned model exists yet. The results do
not choose the primary; this entry does.
*The ablation arm* is the whole train pool (the 6,000 messages of D-030), labelled by
`llm-labeller-v3` on the pinned endpoint (`crusoe/bf16`), with rows tagged `source=llm_v3`, under
the same $4.75 stop. The gate did not accept v3, so it is reported only as a clearly labelled
ablation, never as the primary. It differs from the primary in two ways: label source and size
(6,000 messages against 1,000). The comparison is therefore between the two ways the D4 training
data could be made, not a controlled test of label source alone. The 1,000 hand-labelled messages
are a subset of the pool, so the LLM's agreement with Yash on them is also reported, as a
measurement only.
*The same protocol for both arms.* Each arm trains one multi-task checkpoint, with the same D1–D3
data, the same hyperparameters (`TrainSettings`) and the same seeds; only the D4 training rows
differ. Each is temperature-calibrated on calib (calib v2 for D4) and evaluated once on the frozen
test sets (test v2 for D4) with bootstrap 95% intervals. Both are reported next to the majority
and stock-Laya rows.
*The training sample* (drawn 2026-09-29, before labelling). Seed **20260929**
(`DataSettings.d4_train_hand_seed`); **1,000** messages (`d4_train_hand_rows`), drawn uniformly
from the 6,000-message D-030 pool with `random.Random(f"{seed}:train-hand-v2")`, and shown in that
draw's order. Every message is train-split, from 1,000 distinct threads. None of those threads,
and none of their normalized texts, appear in test v2, calib v2 or any of the 600 D4 eval
candidates; the pool drops text matches, and `draw_train_sample` refuses shared threads. By stratum:
plain/opener 620, cue/opener 241, plain/reply 108, cue/reply 31. That is the pool's natural mix,
unlike the equal strata of test and calib, so its escalation rate will be lower. The sample is
committed in `data/labels/escalate.train_sample.v2.jsonl`; labels go to
`escalate.train_labels.v2.jsonl`.

**D-035 · 2026-09-29 · supersedes D-034's choice of primary · declared before training** — **The
primary D4 arm is the llm_v3 labels. The hand-labelling arm is removed.** (Yash's decision.)
*Why.* Yash abandoned hand-labelling the D4 training sample before labelling any of it: no
`escalate.train_labels.v2.jsonl` exists, and no fine-tuned D4 result exists either, since no
training run has happened. D-034 is left as written. The `label escalate-train` tool and the drawn
sample (`escalate.train_sample.v2.jsonl`) stay in the repository, unused.
*What the primary now learns from.* The D-030 train pool of 6,000 messages, labelled by
`llm-labeller-v3` on Crusoe bf16 (`source=llm_v3`): **899 escalate (15.0%)** and 5,101 not. The
gate did not accept these labels. Against Yash's calib v2 labels, v3 had kappa 0.420, escalate
precision 0.952 and recall 0.426 (D-033). The model therefore learns a stricter notion of
escalation than his, and its recall on his test labels is expected to suffer. `build.TRAIN_FILES`
names `escalate.train.llm_v3.jsonl.gz` as D4's training file.
*The prior shift.* The training pool follows the natural mix (15.0% escalate by v3's labels).
Calib v2 (47 of 100) and test v2 (89 of 200) were drawn in equal strata. Temperature scaling cannot
move a prior; it rescales the logits but adds no offset. So the shift is handled in training:
- **Each epoch draws D4 rows class-balanced**, 50/50 (`TrainSettings.balance_classes`). With the
  5,000-row D4 quota that is 2,500 escalate rows, so each of the 899 is repeated about 2.8 times,
  and 2,500 of the 5,101 not-escalate rows, subsampled.
- Sampling rather than a weighted loss keeps the RLCD loss (policy gradient plus soft
  cross-entropy) untouched, and it is equivalent in expectation to class weights.
- The residual gap between 50% and calib's 47% is left to the threshold chosen on calib.
- The strata mix also differs (the pool is 25% cue messages, test and calib 50%), and that is not
  corrected.
*The escalate threshold.* p(escalate) ≥ θ predicts escalate. θ is chosen on **calib v2** by
maximising macro-F1 over a grid of 0.000 to 1.000 in steps of 0.005; when several grid points
tie, the median of them is taken. For the fine-tuned model, θ is chosen on its calib
probabilities after temperature scaling. For stock Laya, it is chosen on its calib probabilities
at its shipped temperature, the same way, so the gain from fine-tuning can be separated from the
gain from thresholding. The chosen values are logged here once measured. ECE and Brier are
properties of the probabilities, so a threshold does not change them.
*Evaluated once on test v2, with bootstrap 95% intervals:*
- majority;
- stock Laya at the default threshold (argmax);
- stock Laya at its calib threshold;
- the fine-tuned model (llm_v3) with temperature, at its calib threshold.
The same single pass also gives Table A's standard fine-tuned rows at argmax (T = 1 and with
temperature), and the multi-task checkpoint's D1–D3 rows. Every training setting other than D4's
data and its class balancing is `TrainSettings` as committed, including the seed.

**D-035 amendment 1 · 2026-09-29 · amends D-035's "TrainSettings as committed"** — **The first
full training run broke in epoch 2 and was stopped. The policy-gradient term is switched off
(`rl_weight = 0`), and the kept checkpoint is the epoch with the lowest dev loss.** (Yash's
decision, after the diagnosis below.)
*What happened.* Run r1 (`checkpoints/main`, code `832029c`) is kept locally as evidence, and its
log is committed at `results/training/r1_stopped/train_log.jsonl`. It was stopped at update 3,110
of 3,144.
- Training cross-entropy fell through epoch 1, from 0.56 to 0.35.
- It stayed normal for the first ~60 updates of epoch 2, then rose from 0.33 to 1.19 between
  updates 840 and 900, and never recovered.
- Mean cross-entropy by epoch (1, 2, 3):

  | Task | Epoch 1 | Epoch 2 | Epoch 3 |
  |---|---|---|---|
  | D1 turn_complete | 0.602 | 1.831 | 1.976 |
  | D2 barge_in | 0.281 | 0.313 | 0.315 |
  | D3 route | 0.103 | 0.063 | 0.036 |
  | D4 escalate | 0.502 | 0.828 | 0.595 |
*Ruled out.*
- **The LR schedule.** It is cosine with no warmup: LR peaks at update 0 and was a smooth
  2.1e-5 at the break.
- **Sampling.** The sampler, the task mixture and D4's class balance are redrawn only at an epoch
  boundary (update 786), and the break came more than 60 updates later.
- **The epoch-end checkpoint save.** It writes a half-precision copy of the weights and leaves
  the model untouched.
*What it was.* `uv run python scripts/diagnose_r1.py` writes
`results/training/r1_stopped/diagnosis.json`, from train rows only.
- **Pre-clip gradient norms on the stock weights** (median of six micro-batches of eight rows).
  The policy-gradient term against the cross-entropy term:

  | Task | at σ 0.4 | at σ 0.1 |
  |---|---|---|
  | D1 | 2.0× | 10.7× |
  | D2 | 3.9× | 11.4× |
  | D3 | 14.5× | 45× |
  | D4 | 2.0× | 10.7× |

  The term's gradient scales as 1/σ, so annealing σ from 0.4 to 0.1 made it louder every epoch.
- **After epoch 3,** the cross-entropy gradient was 5e-7 to 9e-4, while the policy-gradient
  term's reached 517 on D1.
- **D1 on 400 train rows.** At epoch 3 the model was 85.5% accurate at mean confidence 0.995,
  with p(true) essentially 0 or 1; stock Laya is 45.5% at 0.52. The break is saturated
  overconfidence, not lost discrimination: D1's wrong answers, made at near-certainty, dominate
  its cross-entropy.
- **The mechanism.** RLCD normalises advantages to unit size, so the policy-gradient term stays
  full-size however small the reward differences get. For a right answer, more certainty always
  scores slightly higher under a proper scoring rule, so the term keeps pushing logits apart
  after the cross-entropy gradient has faded. Clipping at 1.0 cannot stop it, because Adam's step
  size does not depend on the gradient's scale.
*The fix.* `TrainSettings.rl_weight = 0.0`: the loss is soft cross-entropy only. The
policy-gradient term is left out of the graph, and its reward is still logged. Calibration stays
the per-decision temperature fitted on calib. This departs from Laya's notebook, which weights the
term at 1; `rl_weight = 1.0` reproduces it. Not chosen:
- **A lower LR:** there is no LR event to fix, and it would only slow the drift.
- **Tighter clipping:** it does not bound Adam's step.
- **A down-weighted term:** there is no principled coefficient, and Adam amplifies any remaining
  term once cross-entropy saturates.
- **Delaying the term:** it is strongest in the late, low-σ epochs, so this runs the wrong way.
*Safeguards for every run from now on.*
- **A checkpoint every epoch** (`checkpoint_epoch{N}`).
- **A dev split carved from train.** It is 5% of each task's train conversations
  (`TrainSettings.dev_fraction`), drawn with a seed and conversation-disjoint from the rows
  trained on, and never calib or test. Those conversations leave training.
- **Best-epoch selection.** After every epoch, dev cross-entropy at T = 1 is scored per task and
  averaged equally over the four tasks. The lowest wins (the earliest on a tie), and that
  checkpoint is copied to the run root, where calibration (temperatures and D4's threshold) and
  evaluation read it.
- **More logging.** Every 10 updates, the log records the pre-clip gradient norm (mean and max)
  and cross-entropy per task.
Everything else is unchanged: learning rates, epochs, quotas, D4's class balancing and the seed.
*Runs.* r1 stays in `checkpoints/main` and is not evaluated. The rerun, r2, goes to
`checkpoints/main-r2`. Before launch, the new path was smoke-tested end to end on the real model:
2 epochs of 64 rows per task, through dev scoring, per-epoch saves, the best-epoch copy and
calibration.

**D-035 outcome · 2026-09-29** — **r2 trained cleanly; D4's calib threshold is 0.505 for both the
fine-tuned model and stock Laya. The fine-tuned model was evaluated once on test v2.**
*Training (r2).* `checkpoints/main-r2`, code `00fac02`, with the policy-gradient term off.
- Dev cross-entropy, averaged over the four tasks, by epoch: 0.218, **0.204**, 0.250, 0.313.
  Epoch 2 is kept.
- From epoch 3, D4 overfits its llm_v3 rows: train cross-entropy fell 0.425 → 0.168 → 0.046 →
  0.012, while dev cross-entropy went 0.312 → 0.281 → 0.485 → 0.707.
- Temperatures fitted on calib: D1 1.081, D2 1.197, D3 1.923, D4 **4.86**. D4's calib NLL went
  from 1.169 to 0.571; the llm_v3-trained model was very overconfident against Yash's calib v2
  labels. No temperature is at a bound.
*The escalate thresholds, chosen on calib v2* (grid 0.005, median of ties):
- the fine-tuned model with its temperature: **θ = 0.505**, calib macro-F1 0.755, 3 grid points
  tied;
- stock Laya at its shipped temperature: **θ = 0.505**, calib macro-F1 0.591, 2 tied.
*D4 on test v2*, n = 200, with bootstrap 95% intervals:

| Row | θ | Accuracy | Macro-F1 | ECE | Brier | Escalate P / R |
|---|---|---|---|---|---|---|
| majority (calib prior) | – | 0.555 [0.485, 0.625] | 0.357 [0.327, 0.385] | 0.025 [0.000, 0.095] | 0.495 [0.487, 0.504] | – / 0.000 |
| stock Laya, default | argmax | 0.665 [0.600, 0.730] | 0.625 [0.554, 0.693] | 0.092 [0.057, 0.159] | 0.422 [0.396, 0.447] | 0.739 / 0.382 |
| stock Laya, calib θ | 0.505 | 0.645 [0.580, 0.710] | 0.579 [0.506, 0.649] | 0.092 [0.057, 0.159] | 0.422 [0.396, 0.447] | 0.781 / 0.281 |
| fine-tuned llm_v3 + T, calib θ | 0.505 | 0.705 [0.640, 0.765] | 0.681 [0.612, 0.747] | 0.064 [0.040, 0.135] | 0.418 [0.363, 0.475] | 0.768 / 0.483 |

- **Thresholding adds almost nothing.** Both thresholds landed next to 0.5: the change is +0.009
  macro-F1 for the fine-tuned model and −0.046 for stock Laya.
- **The gain is from fine-tuning:** +0.056 macro-F1 over stock Laya at its default, and +0.102
  over stock Laya at its own calib threshold.
- **With n = 200 the unpaired intervals overlap,** so these rows alone do not establish the
  difference. A paired bootstrap over the same resamples would.
- **Recall is 0.483 against Yash's labels,** as expected from training labels that under-escalate
  (v3 recall 0.426 on calib).
*D1–D3 from the same single pass* (fine-tuned + temperature, on their v1 test sets):

| Decision | Accuracy | Macro-F1 | ECE | Stock Laya accuracy / macro-F1 |
|---|---|---|---|---|
| D1 | 0.820 [0.813, 0.826] | 0.813 [0.807, 0.820] | 0.010 | 0.488 / 0.485 |
| D2 | 0.969 [0.966, 0.972] | 0.937 [0.931, 0.943] | 0.003 | 0.368 / 0.238 |
| D3 | 0.948 [0.937, 0.959] | 0.906 [0.881, 0.926] | 0.011 | 0.934 / 0.866 |

Hard subsets: D1 0.725 [0.693, 0.755] and D2 0.977 [0.972, 0.981].

**D-036 · 2026-09-29 · §12 Table A · added after the main test pass** — **Paired comparisons from
saved predictions, and two sanity baselines. Nothing here is tuned on test.** (Yash's decisions.)
*Recovering r2's test predictions.* The fine-tuned rows (D-035 outcome) were scored without saving
per-row predictions, and a paired comparison needs them. They were recovered by one inference-only
pass of the same checkpoint over the same frozen test sets (`floorcall eval recover-predictions`).
That pass writes nothing to `results/`, and its logits are used only if every committed metric of
every fine-tuned Table A row reproduces from them identically: confusion, intervals and all.
Otherwise the pass stops and shows the difference. From now on every scoring pass saves its
per-row logits (`runs/eval/`, gitignored and rebuilt by rerunning), so later comparisons never
need another pass.
*Paired bootstrap* (`floorcall eval paired`): fine-tuned against stock Laya on the same resampled
rows, 10,000 resamples, seed 20260927.
- For D4, the fine-tuned model with temperature at its calib threshold, against stock Laya at
  argmax and against stock Laya at its own calib threshold.
- For D1–D3, the fine-tuned model with temperature, against stock Laya.
- Reported per metric: the difference, its 95% percentile interval, and the share of resamples in
  which the fine-tuned model is not better.
- Each model is first checked against its committed Table A row.
*Sanity baselines* (`floorcall eval cheap-baselines`), added after the main test pass. Each is
fitted without test and scored once on test.
- **TF-IDF + logistic regression, all four decisions.** Features are the user's word 1- and
  2-grams and the context's words, through the same normalizer as everything else.
  - It trains on the same rows as r2: each task's train rows minus the same dev split.
  - C is chosen on that dev split by cross-entropy over the grid 0.01, 0.1, 1, 10 and 100.
  - D4 uses balanced class weights, the counterpart of D-035's class-balanced sampling, and its
    threshold is chosen on calib v2 by the D-035 rule.
- **A lexical rule for D2**, fixed before it was scored after looking only at train rows.
  - No words, or fillers only (um, uh, er, ah): noise.
  - At most three words, all from {uh, huh, mhm, hm, hmm, mm, yeah, yep, yes, right, okay, ok,
    oh, sure, really, wow, i, see, true, exactly}: backchannel.
  - Anything else: interruption.
  - Each branch's probabilities are the train class frequencies of the rows taking it
    (add-one smoothed).
*Latency.* Table B is re-measured on the fine-tuned checkpoint (`checkpoints/main-r2`). Its rows
replace the stock-checkpoint rows, which stay in git history.
*D-036 outcome (2026-09-29).*
- **Recovery.** The inference-only pass reproduced every committed fine-tuned Table A row
  identically for all four decisions (metrics, bootstrap intervals and confusion), so its logits
  were used.
- **Paired bootstrap, fine-tuned minus stock** (`results/paired/`). Macro-F1 difference [95%]
  and the share of resamples in which fine-tuning is not better:

  | Decision | Compared with | Macro-F1 difference | Not better |
  |---|---|---|---|
  | D4 | stock Laya at argmax | +0.055 [−0.050, +0.160] | 15.2% |
  | D4 | stock Laya at its calib threshold | +0.101 [−0.003, +0.206] | 2.9% |
  | D1 | stock Laya | +0.329 [+0.318, +0.339] | 0.0% |
  | D2 | stock Laya | +0.699 [+0.690, +0.707] | 0.0% |
  | D3 | stock Laya | +0.039 [+0.016, +0.064] | 0.0% |

  On 200 messages, D4's gain over stock Laya is not established.
- **Sanity baselines** (macro-F1, test): TF-IDF + LR scores D1 0.544, D2 0.850, D3 0.848 and D4
  0.681; the D2 lexical rule scores 0.908. On D4 the bag-of-words model trained on the same
  llm_v3 labels matches the fine-tuned model (0.681). D3's C was chosen at the top of the fixed
  grid (100); the grid is not widened after scoring.

**D-037 · 2026-09-29 · §12 Table B, §14 · the crash, the overclock, and rules for GPU work** —
(Yash's decisions.)
*The crash.* At 18:43 the laptop blue-screened: bugcheck `0x116` (VIDEO_TDR_FAILURE, the display
driver failed to recover from a GPU timeout) and a WHEA "fatal hardware error", then it rebooted
at 18:46. The crash dump is `C:\Windows\Minidump\092926-15921-01.dmp`. Yash traced it to heat: the
laptop reached 95 °C. It came during the CPU rows of the latency run, after the six GPU rows had
finished, and after many hours of sustained GPU load that day (training r2, evaluation, recovery).
No committed result was lost.
*The overclock.* The GPU (RTX 5070 Ti Laptop, driver 591.86) carried a **+150 MHz core /
+150 MHz memory overclock**. It has been removed, and the card now runs at stock clocks. It was in
place for the GPU measurements so far, so:
- **Every GPU latency number so far is discarded:** Table B's GPU rows from 28 September (stock
  checkpoint), and the six GPU rows measured on the fine-tuned checkpoint today, which were never
  committed. The committed rows leave `results/table_b/` and stay in git history. The spike's GPU
  numbers (`docs/spike-m0.md`, D-012) were also taken overclocked and are superseded.
- The discarded fine-tuned rows had come in about a third faster than the stock rows of the day
  before (user_pause with CUDA graphs: p50 29.9 against 45.2 ms). So "latency does not depend on
  the weights" is no longer assumed; stock and fine-tuned are measured side by side.
- Training r1 and r2 and every evaluation also ran overclocked. Their results are numbers, not
  timings, and r2's test logits were reproduced identically in a second pass (D-036), so there is
  no sign of a compute error. They stand. Training throughput figures (rows/s) are overclocked
  figures.
*Rules for the rest of the project.*
- **No GPU work until Yash says go.**
- **GPU latency rows only for now,** at stock clocks on a cool machine. Stock and fine-tuned run
  back to back, interleaved row by row.
  - Each row starts only once the GPU is at or below `EvalSettings.latency_start_max_temp_c`.
  - Telemetry is logged with every row: `nvidia-smi` sampled throughout warmup and timing, giving
    temperature, SM and memory clocks, power draw, power limit, and the clock-event (throttle)
    reasons.
  - A row taken while throttling is **discarded**: any thermal slowdown, hardware slowdown or
    power-brake event during its timed window. It is kept under `runs/latency/discarded/` for the
    record and never written to `results/`. The software power cap is recorded but is not a
    discard reason, because a laptop GPU runs against its power limit whenever it is busy.
  - A row that reaches `latency_abort_temp_c` stops at once.
- **CPU latency rows later, in chunks of at most 20 minutes, with cooldowns between them.** One
  CPU row (1,050 calls at 1–2.6 s) is longer than 20 minutes, so its timed calls are split into
  chunks. Each chunk has its own short warmup, and a cooldown precedes the next chunk.
  Percentiles are computed over all the timed calls. Until they are re-measured, Table B's CPU
  rows remain the stock checkpoint's, from 28 September; they involved no GPU.
- **Table D ablations run on Kaggle, started by Yash.** The notebook, the dataset bundle and the
  upload steps are in `docs/runbook-kaggle.md`. Nothing is started locally.
*D-037 addendum (2026-09-29): Table D on Kaggle.* Kaggle's T4 and P100 GPUs have no fast bf16, so
arms trained there run in fp16 with dynamic loss scaling (`TrainSettings.amp_dtype = "fp16"`, set
by the notebook when the GPU's compute capability is below 8). Laya likewise serves those GPUs in
fp16. That is a different precision on different hardware from r2 (bf16, RTX 5070 Ti), so the
runbook recommends a Kaggle `full` arm as Table D's reference: every Table D row then comes from
the same conditions, and Table A's fine-tuned row stays r2. Whether 4 epochs fit a 12-hour
Kaggle session is measured by the notebook's smoke run. Whatever epoch count is chosen applies to
every arm and is logged before the first full arm starts. Kaggle results cite the bundle's commit
as their code (`FLOORCALL_CODE`, `<commit>-kaggle`). A `max_train_rows_per_task` cap exists for
the smoke run only; its default, None, leaves every run unchanged.

**D-038 · 2026-09-29 · §12 Table D · decided before any Kaggle run** — **Table D is four Kaggle
arms, `full` among them, each trained for 2 epochs.** (Yash's decisions.)
- **A `full` arm on Kaggle is Table D's reference row.** The three ablations (no normalization, no
  `recent_turns`, no `agent_last_utterance`) are read against it, never against r2. All four then
  share precision (fp16 with loss scaling on T4/P100), hardware and recipe. Table A's fine-tuned
  row stays r2.
- **2 epochs for every Kaggle arm, full and ablations alike**, decided now because r2's dev loss
  chose epoch 2 of 4. The best checkpoint is still the epoch with the lowest dev cross-entropy,
  chosen as in D-035 amendment 1, from the two. The cosine learning-rate schedule spans each run's
  own 2 epochs, so a Kaggle arm is not r2 stopped halfway; it is the same recipe at 2 epochs. The
  notebook's `EPOCHS` default is now 2, and its smoke-run time estimate is for information only.
- **The Kaggle dataset is private.** It is derived from Customer Support on Twitter
  (CC BY-NC-SA 4.0), SwDA (CC BY-NC-SA 3.0) and CLINC150 (CC BY 3.0), and holds the frozen test
  sets. `kaggle datasets create` is run without `--public`, which leaves a dataset private, and the
  runbook has a check after upload. The bundle script refuses to finish if any file in the bundle
  looks like a secret: a `.env` or key file by name, a known key pattern, or any value from the
  local `.env`, whether in the files' raw bytes or inside the gzipped data.
*D-037 outcome: GPU latency re-measured (2026-09-29 evening, 2026-09-30 UTC; code `7ca1792`).*
- **Conditions.** Stock clocks, and stock and fine-tuned (r2) interleaved, alternating which went
  first. All 12 rows were kept on their first attempt, and nothing was discarded. No row showed a
  thermal or hardware slowdown or a power brake while timed. The GPU peaked at 64–70 °C. The
  software power cap was active in most samples (95 W limit); that is recorded and is not a
  discard reason.
- **Weights make no difference.** Both checkpoints load their weights as fp32 (421.3M
  parameters), and the fine-tuned model times within −6.7% to +2.8% of stock at p50. The earlier
  "one third faster" came from comparing two overclocked sessions on different days, not from the
  weights.
- **Budget.** No GPU row meets p99 ≤ 50 ms. The closest is CUDA graphs on
  `user_speech_during_agent`: fine-tuned p50 43.0 / p99 57.6 ms, stock 42.4 / 55.2. `user_pause`
  with CUDA graphs is p50 32.5 / p99 64.9 ms. The rows, with their telemetry, are in
  `results/table_b/` (fine-tuned) and `results/table_b/stock/`. The CPU rows are still the stock
  checkpoint's from 28 September and will be re-measured in chunks.
*D-036, paired against the best cheap baseline (2026-09-30).* The fine-tuned model is compared
with the strongest cheap baseline of each decision: the lexical rule for D2, and TF-IDF + LR for
D1, D3 and D4 (D4 at the baseline's own calib threshold). The comparison uses the saved per-row
predictions and the same paired protocol, and nothing is re-scored (`floorcall eval paired
--against baselines`). The check against committed rows keeps the confusion exact but now allows
float round-off of at most 1e-12: the baselines saved log-probabilities, and exp(log p) is not
bit-identical to p (up to 4e-15 in ECE, measured).

| Decision | Baseline | Macro-F1 difference | Not better | Other differences |
|---|---|---|---|---|
| D1 | TF-IDF + LR | +0.269 [+0.259, +0.279] | 0.0% | — |
| D2 | lexical rule | +0.029 [+0.022, +0.035] | 0.0% | the rule's ECE is 0.001 against 0.003; not significant |
| D3 | TF-IDF + LR | +0.058 [+0.032, +0.085] | 0.0% | — |
| D4 | TF-IDF + LR | +0.000 [−0.061, +0.060] | 49.8% | better calibrated: ECE −0.103 [−0.153, −0.032], Brier −0.055 [−0.106, −0.004] |

On D4, fine-tuning buys calibration but no ranking over a bag of words trained on the same LLM
labels.
