# Milestone 0 spike (2026-09-27)

Exit criteria (CLAUDE.md §14): latency numbers in hand, and local fine-tuning confirmed or a
documented decision to use the Kaggle notebook. **Both met.**

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU, 12.8 GB, compute capability 12.0, 90 W cap |
| CPU | Intel Core Ultra 9 285H, 16 cores, 31 GB RAM |
| Power | on AC, Windows "Balanced" plan |
| Stack | torch 2.14.0+cu130, transformers 5.17.0, laya 0.3.21 |
| Checkpoint | `convaiinnovations/laya` @ `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`, bf16 autocast |

Reproduce: `uv run python scripts/spike_m0.py` → `results/spike_m0.json`;
`uv run python scripts/spike_m0_train.py` → `results/spike_m0_train.json`.

## 1. Token budget

Measured by `LayaDecider.state_room` with the real tokenizer on the real question schemas:

| Event | Questions | Room for the state |
|---|---|---|
| `user_pause` | turn_complete, route, escalate | **339 tokens** (route's 16 options set the floor) |
| `user_speech_during_agent` | barge_in, escalate | **434 tokens** |

All 16 route options keep a distinct token span inside `head_max_len=192`.

## 2. Zero-shot behaviour: a qualitative look, not a result

20 probes hand-written by the same person reading the output, so no accuracy is computed from
them. What they show is consistent with Laya's own README ("a fast base to specialise, not a
zero-shot decision engine"):

- **turn_complete does not discriminate.** Every probe got p(true) between 0.59 and 0.80,
  including "and then the charge showed up on".
- **barge_in calls everything an interruption**, "uh huh" and "honey can you grab the door"
  included.
- **route already works** on clear requests (balance, transfer, report_fraud, bill_due, routing,
  pin_change) and flags weather and restaurant requests as out_of_scope. Its confidences read 1.00
  because the checkpoint's `choice:11+` temperature is 0.1006. Laya clamps it to 0.5, which still
  *sharpens* (T < 1), so stock route probabilities are overconfident by construction.
  **Correction (2026-09-28): that last inference was wrong.** It was reasoned from "T < 1
  sharpens", not measured. Measured on the full D3 test set (`results/curves/stock_laya.json`),
  the raw logits are *under*confident, and sharpening them with T = 0.5 lowers ECE from 0.153 to
  0.030. The shipped temperature helps. What remains true is that the unclamped 0.1006 would have
  sharpened far past that.
- **escalate is noisy**: 0.84 for "can i just talk to a real person please", but also 0.78 for
  "when is my electric bill due".

## 3. Latency

Quick look only: 20 warmup / 200 timed on GPU, 5 / 50 on CPU. Table B uses 50 / ≥1,000 through
`floorcall.evaluate.latency`.

| Device | Event | Mode | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|
| cuda | user_pause | one batched call | 57.7 | 77.8 | 82.3 |
| cuda | user_pause | 3 sequential calls | 153.0 | 206.6 | 226.5 |
| cuda | user_speech_during_agent | one batched call | 49.7 | 67.4 | 72.9 |
| cuda | user_speech_during_agent | 2 sequential calls | 108.7 | 129.6 | 132.7 |
| cpu | user_pause | one batched call | 1164.8 | 1209.8 | 1229.0 |
| cpu | user_pause | 3 sequential calls | 797.6 | 833.7 | 840.1 |
| cpu | user_speech_during_agent | one batched call | 427.3 | 442.7 | 450.4 |
| cpu | user_speech_during_agent | 2 sequential calls | 446.9 | 462.2 | 463.9 |

Against the design targets (GPU p99 ≤ 50 ms, CPU p99 ≤ 100 ms), **both paths miss as shipped.**

**GPU: the eager forward is launch-bound.** An exploratory profile (not committed; it reaches
into laya internals, which only the adapter may do) split a `user_pause` call into roughly 2 ms
tokenize, 47 ms forward and 0.6 ms decode. A single 128-token row still took 39 ms. ModernBERT-large
at that size is on the order of 100 GFLOPs, a few milliseconds of real compute, so the time goes to
dispatching around a thousand small kernels one at a time from Python. The same exploration
captured the forward as a **CUDA graph**. Its logits were bit-identical to eager (max |Δ| = 0.0),
and `user_pause` went to p50 28.4 / p99 30.6 ms. Under sustained replay the card sat at 1.9 GHz,
89 W and 99% utilisation, so the remaining ~28 ms is real compute at this laptop's power limit.
These numbers are to be reproduced by the M2 latency harness before any of them is reported;
see DECISIONS.md D-012.

**Batching helps on GPU and hurts on CPU.** One batched call beats sequential calls by 2.2–2.6x on
GPU. On CPU the batched `user_pause` is *slower* (1165 vs 798 ms p50). Every row pads to the longest
one (route, 223 tokens), which triples the token count of the other two rows. A GPU hides that
padding in parallel work; a CPU pays for every token. This is D-003 made concrete.

**CPU is compute-bound** at roughly 490 GFLOPs per `user_pause` call. See D-014.

## 4. Local fine-tuning fits

8 real RLCD updates (the notebook's objective, `floorcall.train.rlcd`) on 512 CLINC150 banking
rows. Half were padded with conversation history to the full 512-token row, so memory is measured
at the longest input training will ever see:

| | |
|---|---|
| peak allocated / reserved | **8.41 GB / 9.09 GB** of 12.8 GB |
| setup | micro-batch 8, accumulation 8 (64 rows per update), GRPO group 4, bf16, gradient checkpointing on encoder and head |
| throughput | 18.3 rows/s overall; 4.5 s per update on 512-token rows, 2.2 s on short rows |

A full run of about 90k rows x 4 epochs projects to roughly 5–6 hours locally, so the Kaggle
notebook is not needed. The loss fell during the run, but that is not a result: the rows were
sorted longest-first, and nothing was held out.
