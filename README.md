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

> **Status: under construction.** Every number below is `TODO` until it has been measured by a
> committed command. Nothing in this README is an estimate.

## Why a decision model and not an LLM

A voice agent has well under a second to hear the user, decide, think and speak before the
conversation feels broken. Speech recognition, the response LLM and speech synthesis already use
most of that budget. The inline decisions (stop talking? respond yet?) have to fit in the tens of
milliseconds that are left, and a generative LLM call cannot.

## Results

### Table A: quality per decision

| Decision | Model | Accuracy | Macro-F1 | ECE | Brier | Hard-subset acc. |
|---|---|---|---|---|---|---|
| D1 turn_complete | majority class | TODO | TODO | TODO | TODO | TODO |
| D1 turn_complete | stock Laya | TODO | TODO | TODO | TODO | TODO |
| D1 turn_complete | fine-tuned | TODO | TODO | TODO | TODO | TODO |
| D1 turn_complete | fine-tuned + temperature | TODO | TODO | TODO | TODO | TODO |
| D2 barge_in | majority class | TODO | TODO | TODO | TODO | TODO |
| D2 barge_in | stock Laya | TODO | TODO | TODO | TODO | TODO |
| D2 barge_in | fine-tuned | TODO | TODO | TODO | TODO | TODO |
| D2 barge_in | fine-tuned + temperature | TODO | TODO | TODO | TODO | TODO |
| D3 route | majority class | TODO | TODO | TODO | TODO | TODO |
| D3 route | stock Laya | TODO | TODO | TODO | TODO | TODO |
| D3 route | fine-tuned | TODO | TODO | TODO | TODO | TODO |
| D3 route | fine-tuned + temperature | TODO | TODO | TODO | TODO | TODO |
| D4 escalate | majority class | TODO | TODO | TODO | TODO | TODO |
| D4 escalate | stock Laya | TODO | TODO | TODO | TODO | TODO |
| D4 escalate | fine-tuned | TODO | TODO | TODO | TODO | TODO |
| D4 escalate | fine-tuned + temperature | TODO | TODO | TODO | TODO | TODO |

Brier is the multi-class form Σₖ(pₖ − yₖ)², range [0, 2]. For a binary decision it is twice the
familiar (p − y)². ECE uses 15 equal-width bins over the probability of the reported answer.

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
