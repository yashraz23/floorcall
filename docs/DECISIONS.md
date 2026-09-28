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
