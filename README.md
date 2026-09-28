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

> **Status, 2026-09-27.** Milestone 0 (the feasibility spike, [docs/spike-m0.md](docs/spike-m0.md))
> is done. So is milestone 1 (data, [docs/data.md](docs/data.md)), apart from the hand-labelled D4
> test set, which is being labelled. Table A's baseline rows for D1–D3 are measured. Every other
> cell says TODO until a committed command measures it, and nothing in this README is an estimate.

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
| D2 barge_in | majority class (train prior) | 0.501 [0.493, 0.510] | 0.223 | 0.010 | 0.564 | 0.453 (5012) |
| D2 barge_in | stock Laya, zero-shot | 0.368 [0.360, 0.376] | 0.238 | 0.005 | 0.661 | 0.439 (5012) |
| D2 barge_in | fine-tuned | TODO | TODO | TODO | TODO | TODO |
| D2 barge_in | fine-tuned + temperature | TODO | TODO | TODO | TODO | TODO |
| D3 route | majority class (train prior) | 0.690 [0.666, 0.713] | 0.051 | 0.547 | 0.837 | n/a |
| D3 route | stock Laya, zero-shot | 0.934 [0.921, 0.946] | 0.866 | 0.030 | 0.115 | n/a |
| D3 route | fine-tuned | TODO | TODO | TODO | TODO | TODO |
| D3 route | fine-tuned + temperature | TODO | TODO | TODO | TODO | TODO |
| D4 escalate | majority class (train prior) | TODO | TODO | TODO | TODO | TODO |
| D4 escalate | stock Laya, zero-shot | TODO | TODO | TODO | TODO | TODO |
| D4 escalate | fine-tuned | TODO | TODO | TODO | TODO | TODO |
| D4 escalate | fine-tuned + temperature | TODO | TODO | TODO | TODO | TODO |
<!-- table-a:end -->

Brier is the multi-class form Σₖ(pₖ − yₖ)², range [0, 2]; for a binary decision it is twice the
familiar (p − y)². ECE uses 15 equal-width bins over the probability of the reported answer. The
95% interval on accuracy is Wilson's. D1's hard subset is truncations only, all labelled
incomplete, so a constant "incomplete" predictor such as the majority baseline scores 1.000 on it;
the column is only informative for models that are not constant. D3's test set is 69%
out_of_scope, so macro-F1 is its headline number.

### Table B: latency (batch 1)

| Path | p50 | p95 | p99 | Fits budget |
|---|---|---|---|---|
| floorcall, GPU (RTX 5070 Ti Laptop) | TODO | TODO | TODO | TODO |
| floorcall, CPU | TODO | TODO | TODO | TODO |
| 3 questions, one batched call | TODO | TODO | TODO | TODO |
| 3 questions, 3 sequential calls | TODO | TODO | TODO | TODO |

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
