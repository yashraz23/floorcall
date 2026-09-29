# floorcall

**A millisecond decision layer for voice agents.** Inside a real-time voice pipeline, floorcall
answers the typed questions an agent must settle before it acts, in one batched forward call of a
small encoder model:

- Is the user done talking, or just pausing?
- The user spoke while the agent was talking. Was that "uh-huh", a real interruption, or noise?
- What does the user want, and does this agent handle it at all?
- Should this conversation go to a human?

It runs [Laya](https://github.com/NandhaKishorM/laya), an open-weight (Apache-2.0),
non-autoregressive decision model, fine-tuned and recalibrated on real conversational data. It is
plugged into a [Pipecat](https://github.com/pipecat-ai/pipecat) voice pipeline.

> **Status, 2026-09-29.** Milestone 0 (the feasibility spike, [docs/spike-m0.md](docs/spike-m0.md))
> is done. So is milestone 1 (data, [docs/data.md](docs/data.md)), including D4's hand-labelled
> test set: v2, 200 messages relabelled blind under guideline v2 (docs/DECISIONS.md D-033). Table
> A's baseline rows are measured for all four decisions. Every other cell says TODO until a
> committed command measures it, and nothing in this README is an estimate.

## Why a decision model and not an LLM

A voice agent has well under a second to hear the user, decide, think and speak before the
conversation feels broken. Speech recognition, the response LLM and speech synthesis already use
most of that budget. The inline decisions (stop talking? respond yet?) have to fit in the tens of
milliseconds that are left, and a generative LLM call cannot.

## Results

### Table A: quality per decision

Frozen test sets (`data/test_frozen/`). Rendered from `results/table_a/` by `uv run floorcall eval
readme`; a row without a results file says TODO.

<!-- table-a:start -->
| Decision | Model | Accuracy [95% CI] | Macro-F1 | ECE | Brier | Hard-subset acc. (n) |
|---|---|---|---|---|---|---|
| D1 turn_complete | majority class (train prior) | 0.590 [0.582, 0.597] | 0.371 | 0.000 | 0.484 | 1.000 (775) |
| D1 turn_complete | stock Laya, zero-shot | 0.488 [0.480, 0.496] | 0.485 | 0.028 | 0.510 | 0.505 (775) |
| D1 turn_complete | fine-tuned | TODO | TODO | TODO | TODO | TODO |
| D1 turn_complete | fine-tuned + temperature | TODO | TODO | TODO | TODO | TODO |
| D1 turn_complete | prompted LLM, stated probabilities | TODO | TODO | TODO | TODO | TODO |
| D2 barge_in | majority class (train prior) | 0.501 [0.493, 0.510] | 0.223 | 0.010 | 0.564 | 0.453 (5012) |
| D2 barge_in | stock Laya, zero-shot | 0.368 [0.360, 0.376] | 0.238 | 0.005 | 0.661 | 0.439 (5012) |
| D2 barge_in | fine-tuned | TODO | TODO | TODO | TODO | TODO |
| D2 barge_in | fine-tuned + temperature | TODO | TODO | TODO | TODO | TODO |
| D2 barge_in | prompted LLM, stated probabilities | TODO | TODO | TODO | TODO | TODO |
| D3 route | majority class (train prior) | 0.690 [0.666, 0.713] | 0.051 | 0.547 | 0.837 | n/a |
| D3 route | stock Laya, zero-shot | 0.934 [0.921, 0.946] | 0.866 | 0.030 | 0.115 | n/a |
| D3 route | fine-tuned | TODO | TODO | TODO | TODO | TODO |
| D3 route | fine-tuned + temperature | TODO | TODO | TODO | TODO | TODO |
| D3 route | prompted LLM, stated probabilities | TODO | TODO | TODO | TODO | TODO |
| D4 escalate | majority class (calib prior) | 0.555 [0.485, 0.625] | 0.357 [0.327, 0.385] | 0.025 [0.000, 0.095] | 0.495 [0.487, 0.504] | 0.710 [0.620, 0.800] (100) |
| D4 escalate | stock Laya, zero-shot | 0.665 [0.600, 0.730] | 0.625 [0.554, 0.693] | 0.092 [0.057, 0.159] | 0.422 [0.396, 0.447] | 0.730 [0.640, 0.810] (100) |
| D4 escalate | stock Laya, calib threshold | TODO | TODO | TODO | TODO | TODO |
| D4 escalate | fine-tuned | TODO | TODO | TODO | TODO | TODO |
| D4 escalate | fine-tuned + temperature | TODO | TODO | TODO | TODO | TODO |
| D4 escalate | fine-tuned + temperature, calib threshold | TODO | TODO | TODO | TODO | TODO |
| D4 escalate | prompted LLM, stated probabilities | TODO | TODO | TODO | TODO | TODO |
<!-- table-a:end -->

Brier is the multi-class form Σₖ(pₖ − yₖ)², range [0, 2]; for a binary decision it is twice the
familiar (p − y)². ECE uses 15 equal-width bins over the probability of the reported answer. The
95% interval on accuracy is Wilson's, except in rows scored with a percentile bootstrap (10,000
resamples of the rows): there every metric has its 95% interval in brackets. The D4 rows are
bootstrapped because D4's test set is small: 200 hand-labelled messages, drawn in four equal
strata, so its escalation rate (89 of 200) is not the natural one. D4's majority prior comes from calib,
which is drawn the same way. D1's hard subset is truncations only, all labelled
incomplete, so a constant "incomplete" predictor such as the majority baseline scores 1.000 on it;
the column is only informative for models that are not constant. D3's test set is 69%
out_of_scope, so macro-F1 is its headline number.

**Not compared: LiveKit's text turn detector.** The spec planned it as a D1 baseline, and the
model (`livekit/turn-detector`) is still published. Its licence, the LiveKit Model License, allows
use only "with LiveKit Agents", not "on a standalone basis", and forbids using its outputs "to
improve or otherwise develop any other models". A comparison run in this repository's harness would
be standalone use, so it is not run here (docs/DECISIONS.md D-029).

### Table B: latency (batch 1)

Rendered from `results/table_b/` (`uv run floorcall eval latency`, then `uv run floorcall eval
readme`). Budgets: p99 ≤ 50 ms on GPU, ≤ 100 ms on CPU.

<!-- table-b:start -->
| Path | p50 ms | p95 ms | p99 ms | of which forward, p50 | of which packing, p50 | Fits budget (p99) |
|---|---|---|---|---|---|---|
| GPU, CUDA graphs: user_pause, 3 questions in 1 call | 45.2 | 89.0 | 95.2 | 38.2 | 1.3 | no (≤ 50 ms) |
| GPU, CUDA graphs: user_pause, 3 questions in 3 calls | 53.0 | 89.0 | 93.1 | 40.6 | 1.0 | no (≤ 50 ms) |
| GPU, CUDA graphs: user_speech_during_agent, 2 questions in 1 call | 55.5 | 67.0 | 70.6 | 47.8 | 2.6 | no (≤ 50 ms) |
| GPU, eager: user_pause, 3 questions in 1 call | 72.3 | 97.0 | 104.0 | 65.4 | 1.3 | no (≤ 50 ms) |
| GPU, eager: user_pause, 3 questions in 3 calls | 174.8 | 218.5 | 232.9 | 160.1 | 1.2 | no (≤ 50 ms) |
| GPU, eager: user_speech_during_agent, 2 questions in 1 call | 72.6 | 87.4 | 92.2 | 64.2 | 2.9 | no (≤ 50 ms) |
| CPU: user_pause, 3 questions in 1 call | 1160.9 | 2577.6 | 2647.8 | 1156.7 | 0.7 | no (≤ 100 ms) |
| CPU: user_pause, 3 questions in 3 calls | 822.6 | 2299.0 | 2347.1 | 814.9 | 0.6 | no (≤ 100 ms) |
| CPU: user_speech_during_agent, 2 questions in 1 call | 1351.5 | 1716.0 | 1750.9 | 1347.1 | 1.6 | no (≤ 100 ms) |
| prompted LLM (OpenRouter), user_pause, 3 questions in 1 call, network included | TODO | TODO | TODO | TODO | TODO | TODO |
<!-- table-b:end -->

<!-- table-b-env:start -->
Measured on NVIDIA GeForce RTX 5070 Ti Laptop GPU (driver, power limit: 591.86, [N/A]) and Intel64 Family 6 Model 197 Stepping 2, GenuineIntel with 16 torch threads; on AC power: True; torch 2.14.0+cu130, laya 0.3.21. Batch 1, 50 warmup and 1000 timed iterations over 50 fixed inputs per event; every timed call is a full `Decider.decide` (packing, tokenizing, forward, temperatures).
<!-- table-b-env:end -->

"3 questions in 1 call" is one batched forward pass with one row per question; each row re-reads
the state (docs/DECISIONS.md D-003). Latency depends on the architecture and input lengths, not the
weights, so these numbers hold for the stock and the fine-tuned checkpoint alike.

Each row is one run, and on this laptop the tail moves between runs. Two runs of the GPU rows
(f4c7e87, then ae690c4) differed by 10–20% at p99 in both directions, more than the code change
between them explains (docs/DECISIONS.md D-028). Read a p99 here as one measurement on a
power-limited laptop GPU, not a guarantee.

### Table C: robustness to ASR noise

The fine-tuned, calibrated model on the same test sets, with the user's words degraded as a
recogniser might: each word deleted with probability level/2 or replaced by a common word with
probability level/2, and the last one or two words missing with probability level
(`floorcall.evaluate.robustness`). Cells are macro-F1 (accuracy).

<!-- table-c:start -->
| Decision | noise 0.00 | noise 0.05 | noise 0.10 | noise 0.20 |
|---|---|---|---|---|
| D1 turn_complete | TODO | TODO | TODO | TODO |
| D2 barge_in | TODO | TODO | TODO | TODO |
| D3 route | TODO | TODO | TODO | TODO |
| D4 escalate | TODO | TODO | TODO | TODO |
<!-- table-c:end -->

### Table D: ablations

Each variant is a separately trained, calibrated checkpoint, scored under the same state settings it
trained with. The normalization ablation is scored twice: on written text, where punctuation leaks
the answer, and on ASR-style text, which is what a live pipeline delivers.

<!-- table-d:start -->
| Variant | D1 macro-F1 (acc) | D1 hard acc. | D2 macro-F1 (acc) | D2 hard acc. | D3 macro-F1 (acc) | D4 macro-F1 (acc) |
|---|---|---|---|---|---|---|
| full model | TODO | TODO | TODO | TODO | TODO | TODO |
| without recent_turns | TODO | TODO | TODO | TODO | TODO | TODO |
| without agent_last_utterance | TODO | TODO | TODO | TODO | TODO | TODO |
| without normalization, scored on written text | TODO | TODO | TODO | TODO | TODO | TODO |
| without normalization, scored on ASR-style text | TODO | TODO | TODO | TODO | TODO | TODO |
<!-- table-d:end -->

### Operating points and curves

Each threshold is a point on a tradeoff curve, chosen on the **calib** split: the smallest θ whose
calib rate of the costly error stays at or under 5% (false stops for θ_interrupt, premature
responses for θ_yield). The test columns then show what that θ does. Added delay assumes the pause
event fires after 300 ms of silence and the safety net answers at 2,000 ms
(`floorcall.evaluate.curves`). Rendered from `results/curves/` by `uv run floorcall eval curves`,
then `eval figures` and `eval readme`.

<!-- operating-points:start -->
| Model | θ_interrupt | false stops (test) | missed interruptions (test) | θ_yield | premature responses (test) | added delay, ms (test) |
|---|---|---|---|---|---|---|
| stock Laya | 0.340 | 0.040 | 0.931 | 0.630 | 0.055 | 1599 |
| fine-tuned + temperature | TODO | TODO | TODO | TODO | TODO | TODO |
<!-- operating-points:end -->

<!-- figures:start -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/tradeoff_interrupt.dark.png">
  <img alt="θ_interrupt: false stops against missed interruptions" src="results/figures/tradeoff_interrupt.light.png" width="640">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/tradeoff_yield.dark.png">
  <img alt="θ_yield: premature responses against added delay" src="results/figures/tradeoff_yield.light.png" width="640">
</picture>

*Reliability, fine-tuned model, before and after temperature*: TODO

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figures/reliability_stock_laya.dark.png">
  <img alt="Reliability, stock Laya: raw logits and shipped temperatures" src="results/figures/reliability_stock_laya.light.png" width="640">
</picture>
<!-- figures:end -->

## Design notes

Every divergence from the original spec, and why, is in [docs/DECISIONS.md](docs/DECISIONS.md).

## Setup

```bash
uv sync                                              # NVIDIA GPU (CUDA 13 driver), the default
uv sync --no-default-groups --group dev --group cpu  # no NVIDIA GPU
uv run pytest
```

## Licence

Code: Apache-2.0. Derived datasets inherit their sources' licences, including SwDA's
CC BY-NC-SA 3.0 (non-commercial), and each dataset card says which.
