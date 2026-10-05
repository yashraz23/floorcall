---
license: cc-by-nc-sa-4.0
language:
- en
base_model: convaiinnovations/laya
base_model_relation: finetune
pipeline_tag: text-classification
tags:
- voice-agents
- turn-taking
- barge-in
- intent-classification
- calibration
- laya
metrics:
- accuracy
- f1
---

# floorcall

**A fine-tuned, calibrated decision checkpoint for voice agents.** It answers four typed questions
that a voice agent must settle before it acts, from the conversation's text, in one batched
forward call:

| Event | When it fires | Questions asked in one call |
|---|---|---|
| `user_pause` | voice activity detection hears silence while the user holds the floor | D1 `turn_complete`, D3 `route`, D4 `escalate` |
| `user_speech_during_agent` | the user speaks while the agent's speech is playing | D2 `barge_in`, D4 `escalate` |

- **D1 `turn_complete`** (yes/no): has the user finished their turn, or are they mid-thought?
- **D2 `barge_in`** (`backchannel` / `interruption` / `noise`): was the speech over the agent "uh-huh",
  a real interruption, or not meant for the agent?
- **D3 `route`** (15 banking intents + `out_of_scope`): what does the user want, and does a bank
  support agent handle it?
- **D4 `escalate`** (yes/no): is the user asking for a person, or frustrated and not being helped?

It is [Laya](https://huggingface.co/convaiinnovations/laya) (Convai Innovations; a ModernBERT-large
encoder with a decision head, about 421M parameters), fine-tuned on all four decisions together,
with one temperature per decision fitted on held-out calibration data. The output probabilities are
meant to be thresholded deliberately: each threshold is a point on a measured tradeoff curve.

## Model details

| | |
|---|---|
| Base model | `convaiinnovations/laya`, revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851` (English) |
| Fine-tuning | one multi-task checkpoint, soft cross-entropy only, 4 epochs; epoch 2 kept by the lowest dev cross-entropy (dev = 5% of train conversations, held out) |
| Training run | r2: bf16 on an NVIDIA GeForce RTX 5070 Ti Laptop GPU, 3.4 hours, seed 20260927, code `00fac02` |
| Calibration | one temperature per decision, fitted on each decision's calib split (`floorcall_calibration.json`) |
| Inference precision | fp16 autocast on CUDA (the default), fp32 on CPU |
| Input | a JSON state (below), all text lowercased with punctuation removed |
| Language | English only |
| Licence | CC BY-NC-SA 4.0: non-commercial, share-alike ([Licence](#licence)) |

**Files.** `model.safetensors` (weights), `rl_agent_config.json`, `encoder/config.json` and
`tokenizer/` (Laya's configuration and ModernBERT-large's tokenizer, as Laya ships them), and
`floorcall_calibration.json` (the four temperatures, their calib fits, and θ_escalate), and
`LICENSE-laya-Apache-2.0.txt` (Laya's licence). There is no training, calibration or test data in
this repository.

## How to use it

The model reads a **state**: the user's words so far, the agent's last utterance, and recent
turns. Speech recognisers emit no punctuation or casing, so every text field must be normalized
the same way as in training: lowercase, punctuation removed, whitespace collapsed. A model that
sees punctuation it was not trained with is the "without normalization" row of Table D.

```json
{
  "agent_speaking": true,
  "recent_turns": [
    {"speaker": "user", "text": "i tried to pay at the grocery store"},
    {"speaker": "agent", "text": "i can help with that let me check your card"}
  ],
  "agent_last_utterance": "your card ending 4417 was declined because the",
  "user_partial": "yeah but why"
}
```

### With floorcall (recommended)

The [floorcall code](https://github.com/yashraz23/floorcall) normalizes, packs the state under the
model's token budget, asks the exact question definitions the model was trained with
(`src/floorcall/questions.py`), and applies the calibrated temperatures. It needs Python 3.11 and
`laya==0.3.21`.

```python
from pathlib import Path

from huggingface_hub import snapshot_download

from floorcall.config import LayaSettings, get_settings
from floorcall.decider import Decider
from floorcall.questions import Event
from floorcall.state import Snapshot, Turn

# Download first: the Decider reads floorcall_calibration.json from a local folder.
path = snapshot_download("enz23/floorcall")
settings = get_settings()
settings = settings.model_copy(
    update={"laya": LayaSettings(checkpoint=path, revision=None, device="cpu")}  # or "cuda"
)
decider = Decider.load(settings)

decision = decider.decide(
    Event.USER_SPEECH_DURING_AGENT,
    Snapshot(
        agent_speaking=True,
        agent_last_utterance="Your card ending 4417 was declined because the",
        user_partial="yeah but why",
        recent_turns=(
            Turn("user", "I tried to pay at the grocery store."),
            Turn("agent", "I can help with that, let me check your card."),
        ),
    ),
)
decision.probabilities["barge_in"]  # {'backchannel': 0.002, 'interruption': 0.976, 'noise': 0.021}
decision.probabilities["escalate"]  # {'false': 0.691, 'true': 0.309}
```

### With Laya alone

`laya.Agent(path).predict(state, questions)` loads the folder too, with the same question
dictionaries. Its probabilities are the uncalibrated ones (the "fine-tuned" rows of Table A). To
calibrate, raise each probability to `1 / T`, with `T` the decision's temperature in
`floorcall_calibration.json`, and renormalize: that equals `softmax(logits / T)`. Laya rounds the
probabilities it returns, so this matches floorcall's output to about 1e-4. You must also
normalize the text and keep the state inside the model's window yourself.

## Intended use

- An inline decision layer inside an English voice agent's pipeline, run on transcribed text at
  voice-activity events: when to respond, whether to stop speaking for the user, where to route a
  banking request, and when to offer a human.
- Research and teaching on calibrated decision models, turn-taking and barge-in classification.
- Non-commercial use only (CC BY-NC-SA 4.0).

## Out-of-scope use

- **Commercial use**, under this licence.
- **Decisions about people** (credit, fraud, eligibility, identity). `route` and `escalate` steer a
  conversation, nothing more; `report_fraud` means the user wants to report fraud, not that fraud
  happened.
- **Domains other than retail banking support** for `route`: its 15 intents are CLINC150's banking
  domain, and everything else is `out_of_scope`.
- **Languages other than English**, and text that is not conversational speech.
- **Hard real-time guarantees.** No measured path meets its p99 latency target (Table B).
- **Replacing a human hand-off policy.** D4 is weak (below); use it to offer a person sooner, never
  as the only route to one.

## Results

All numbers are on frozen test sets that no training run, calibration or threshold choice ever
saw. They come from the result files in the floorcall repository and are rendered into this card
by `uv run floorcall release card`. "Fine-tuned" rows are this checkpoint.

### Table A: quality per decision

<!-- table-a:start -->
| Decision | Model | Accuracy [95% CI] | Macro-F1 | ECE | Brier | Hard-subset acc. (n) |
|---|---|---|---|---|---|---|
| D1 turn_complete | majority class (train prior) | 0.590 [0.582, 0.597] | 0.371 | 0.000 | 0.484 | 1.000 (775) |
| D1 turn_complete | TF-IDF + logistic regression | 0.624 [0.616, 0.632] | 0.544 [0.536, 0.553] | 0.008 [0.006, 0.018] | 0.456 [0.452, 0.460] | 0.843 [0.817, 0.868] (775) |
| D1 turn_complete | stock Laya, zero-shot | 0.488 [0.480, 0.496] | 0.485 | 0.028 | 0.510 | 0.505 (775) |
| D1 turn_complete | fine-tuned | 0.820 [0.813, 0.826] | 0.813 [0.807, 0.820] | 0.020 [0.015, 0.026] | 0.243 [0.237, 0.250] | 0.725 [0.693, 0.755] (775) |
| D1 turn_complete | fine-tuned + temperature | 0.820 [0.813, 0.826] | 0.813 [0.807, 0.820] | 0.010 [0.007, 0.016] | 0.243 [0.236, 0.249] | 0.725 [0.693, 0.755] (775) |
| D1 turn_complete | prompted LLM, stated probabilities | 0.721 [0.714, 0.728] | 0.721 | 0.138 | 0.435 | 0.412 (775) |
| D2 barge_in | majority class (train prior) | 0.501 [0.493, 0.510] | 0.223 | 0.010 | 0.564 | 0.453 (5012) |
| D2 barge_in | lexical rule: backchannel words + length | 0.951 [0.948, 0.955] | 0.908 [0.901, 0.916] | 0.001 [0.000, 0.005] | 0.092 [0.085, 0.098] | 0.985 [0.981, 0.988] (5012) |
| D2 barge_in | TF-IDF + logistic regression | 0.931 [0.926, 0.935] | 0.850 [0.840, 0.859] | 0.015 [0.012, 0.019] | 0.102 [0.097, 0.107] | 0.963 [0.958, 0.968] (5012) |
| D2 barge_in | stock Laya, zero-shot | 0.368 [0.360, 0.376] | 0.238 | 0.005 | 0.661 | 0.439 (5012) |
| D2 barge_in | fine-tuned | 0.969 [0.966, 0.972] | 0.937 [0.931, 0.943] | 0.008 [0.006, 0.010] | 0.042 [0.039, 0.046] | 0.977 [0.972, 0.981] (5012) |
| D2 barge_in | fine-tuned + temperature | 0.969 [0.966, 0.972] | 0.937 [0.931, 0.943] | 0.003 [0.003, 0.006] | 0.042 [0.038, 0.045] | 0.977 [0.972, 0.981] (5012) |
| D2 barge_in | prompted LLM, stated probabilities | 0.708 [0.700, 0.716] | 0.574 | 0.196 | 0.493 | 0.711 (5012) |
| D3 route | majority class (train prior) | 0.690 [0.666, 0.713] | 0.051 | 0.547 | 0.837 | n/a |
| D3 route | TF-IDF + logistic regression | 0.908 [0.893, 0.923] | 0.848 [0.820, 0.872] | 0.057 [0.046, 0.071] | 0.147 [0.130, 0.165] | n/a |
| D3 route | stock Laya, zero-shot | 0.934 [0.921, 0.946] | 0.866 | 0.030 | 0.115 | n/a |
| D3 route | fine-tuned | 0.948 [0.937, 0.959] | 0.906 [0.881, 0.926] | 0.043 [0.033, 0.054] | 0.090 [0.071, 0.111] | n/a |
| D3 route | fine-tuned + temperature | 0.948 [0.937, 0.959] | 0.906 [0.881, 0.926] | 0.011 [0.008, 0.023] | 0.082 [0.065, 0.100] | n/a |
| D3 route | prompted LLM, stated probabilities | 0.970 [0.961, 0.979] | 0.939 [0.918, 0.956] | 0.028 [0.021, 0.037] | 0.058 [0.044, 0.074] | n/a |
| D4 escalate | majority class (calib prior) | 0.555 [0.485, 0.625] | 0.357 [0.327, 0.385] | 0.025 [0.000, 0.095] | 0.495 [0.487, 0.504] | 0.710 [0.620, 0.800] (100) |
| D4 escalate | TF-IDF + logistic regression (θ = 0.240) | 0.685 [0.620, 0.750] | 0.681 [0.613, 0.743] | 0.167 [0.125, 0.239] | 0.473 [0.386, 0.561] | 0.740 [0.650, 0.820] (100) |
| D4 escalate | stock Laya, zero-shot | 0.665 [0.600, 0.730] | 0.625 [0.554, 0.693] | 0.092 [0.057, 0.159] | 0.422 [0.396, 0.447] | 0.730 [0.640, 0.810] (100) |
| D4 escalate | stock Laya, calib threshold (θ = 0.505) | 0.645 [0.580, 0.710] | 0.579 [0.506, 0.649] | 0.092 [0.057, 0.159] | 0.422 [0.396, 0.447] | 0.700 [0.610, 0.790] (100) |
| D4 escalate | fine-tuned | 0.695 [0.630, 0.755] | 0.672 [0.602, 0.737] | 0.277 [0.220, 0.340] | 0.544 [0.434, 0.658] | 0.740 [0.650, 0.820] (100) |
| D4 escalate | fine-tuned + temperature | 0.695 [0.630, 0.755] | 0.672 [0.602, 0.737] | 0.064 [0.040, 0.135] | 0.418 [0.363, 0.475] | 0.740 [0.650, 0.820] (100) |
| D4 escalate | fine-tuned + temperature, calib threshold (θ = 0.505) | 0.705 [0.640, 0.765] | 0.681 [0.612, 0.747] | 0.064 [0.040, 0.135] | 0.418 [0.363, 0.475] | 0.750 [0.660, 0.830] (100) |
| D4 escalate | prompted LLM, stated probabilities | 0.635 [0.570, 0.700] | 0.610 [0.538, 0.677] | 0.248 [0.186, 0.312] | 0.539 [0.445, 0.635] | 0.570 [0.470, 0.660] (100) |
<!-- table-a:end -->

Accuracy intervals are Wilson's, except in rows scored with a percentile bootstrap (10,000
resamples), where every metric has its 95% interval in brackets. Brier is the multi-class form,
range [0, 2]. ECE uses 15 equal-width bins. D1's hard subset is truncations ending on a
turn-final word (all "incomplete", so a constant predictor scores 1.000 there). D2's hard subset is
surface forms both classes use ("yeah" vs "yeah but that's not what i asked"). D3's test set is
69% `out_of_scope`, so macro-F1 is its headline number. D4's test set is 200 hand-labelled
messages drawn in four equal strata, so its escalation rate (89 of 200) is not the natural one.

### Table B: latency, batch 1

<!-- table-b:start -->
| Path | p50 ms | p95 ms | p99 ms | of which forward, p50 | of which packing, p50 | Fits budget (p99) |
|---|---|---|---|---|---|---|
| GPU, CUDA graphs: user_pause, 3 questions in 1 call | 29.1 | 55.6 | 59.0 | 24.6 | 0.8 | no (≤ 50 ms) |
| GPU, CUDA graphs: user_pause, 3 questions in 3 calls | 52.4 | 79.2 | 85.2 | 43.1 | 0.8 | no (≤ 50 ms) |
| GPU, CUDA graphs: user_speech_during_agent, 2 questions in 1 call | 44.3 | 54.9 | 57.4 | 38.3 | 1.9 | no (≤ 50 ms) |
| GPU, eager: user_pause, 3 questions in 1 call | 56.4 | 82.3 | 88.5 | 50.3 | 1.1 | no (≤ 50 ms) |
| GPU, eager: user_pause, 3 questions in 3 calls | 120.4 | 158.6 | 179.3 | 110.0 | 0.9 | no (≤ 50 ms) |
| GPU, eager: user_speech_during_agent, 2 questions in 1 call | 55.4 | 66.4 | 73.7 | 49.1 | 2.5 | no (≤ 50 ms) |
| CPU: user_pause, 3 questions in 1 call | 976.9 | 2162.4 | 2219.8 | 973.1 | 0.7 | no (≤ 100 ms) |
| CPU: user_pause, 3 questions in 3 calls | 680.9 | 1899.1 | 1953.6 | 674.0 | 0.6 | no (≤ 100 ms) |
| CPU: user_speech_during_agent, 2 questions in 1 call | 1153.6 | 1426.7 | 1464.5 | 1149.7 | 1.4 | no (≤ 100 ms) |
| prompted LLM (gpt-oss-20b via OpenRouter, pinned DeepInfra bf16), user_pause, 3 questions in 1 call: network latency, request to parsed answer | 2565.1 | 4270.7 | 6072.2 | n/a | n/a | no (≤ 50 ms) |
<!-- table-b:end -->

<!-- table-b-env:start -->
Measured on NVIDIA GeForce RTX 5070 Ti Laptop GPU (driver 591.86) and Intel64 Family 6 Model 197 Stepping 2, GenuineIntel with 16 torch threads; on AC power: True; torch 2.14.0+cu130, laya 0.3.21. GPU rows measured 2026-09-30 (UTC). Windows power mode: Best performance. GPU power limit: vendor default, 80 W base plus Dynamic Boost (115 W enforced at the start). Batch 1, 50 warmup and 1000 timed iterations over 50 fixed inputs per event; every timed call is a full `Decider.decide` (packing, tokenizing, forward, temperatures).
<!-- table-b-env:end -->

Budgets: p99 ≤ 50 ms on GPU, ≤ 100 ms on CPU. Each row is one run on a power-limited laptop GPU;
read a p99 as one measurement, not a guarantee.

### Table C: robustness to ASR-style noise

Each user word deleted or replaced by a common word with probability level/2 each, and the last
one or two words missing with probability level. Cells are macro-F1 (accuracy).

<!-- table-c:start -->
| Decision | noise 0.00 | noise 0.05 | noise 0.10 | noise 0.20 |
|---|---|---|---|---|
| D1 turn_complete | 0.814 (0.820) | 0.775 (0.786) | 0.743 (0.760) | 0.679 (0.713) |
| D2 barge_in | 0.937 (0.969) | 0.865 (0.927) | 0.810 (0.887) | 0.728 (0.814) |
| D3 route | 0.906 (0.948) | 0.882 (0.937) | 0.845 (0.919) | 0.759 (0.878) |
| D4 escalate | 0.672 (0.695) | 0.650 (0.675) | 0.643 (0.680) | 0.631 (0.665) |
<!-- table-c:end -->

### Table D: ablations

Each ablation is trained from Laya again, with part of the state taken away, on Kaggle (Tesla T4,
fp16, 2 epochs). It is compared with a **full** arm trained the same way on the same hardware, not
with this checkpoint. The full arm is only that reference and is not released. The normalization
ablation is scored twice: on written text, which it trained on, and on ASR-style text, which is
what a live pipeline delivers.

<!-- table-d:start -->
| Variant | D1 macro-F1 (acc) | D1 hard acc. | D2 macro-F1 (acc) | D2 hard acc. | D3 macro-F1 (acc) | D4 macro-F1 (acc) |
|---|---|---|---|---|---|---|
| full model | 0.824 (0.827) | 0.675 | 0.941 (0.973) | 0.994 | 0.911 (0.955) | 0.695 (0.715) |
| without recent_turns | 0.823 (0.826) | 0.717 | 0.941 (0.971) | 0.982 | 0.924 (0.958) | 0.629 (0.680) |
| without agent_last_utterance | 0.820 (0.823) | 0.720 | 0.942 (0.971) | 0.987 | 0.929 (0.964) | 0.648 (0.685) |
| without normalization, scored on written text | 0.959 (0.960) | 0.992 | 0.945 (0.972) | 0.981 | 0.925 (0.957) | 0.695 (0.715) |
| without normalization, scored on ASR-style text | 0.381 (0.593) | 1.000 | 0.938 (0.969) | 0.979 | 0.927 (0.959) | 0.701 (0.720) |
<!-- table-d:end -->

Each variant minus the full arm on the same test rows, with paired-bootstrap 95% intervals
(10,000 resamples; hard-subset accuracy resamples hard rows only). D4 is at argmax, as Table D
scores it.

<!-- table-d-paired:start -->
| Variant minus full arm | D1 macro-F1 | D1 hard acc. | D2 macro-F1 | D2 hard acc. | D3 macro-F1 | D4 macro-F1 |
|---|---|---|---|---|---|---|
| without recent_turns | -0.001 [-0.006, +0.003] | +0.043 [+0.021, +0.066] | -0.000 [-0.007, +0.005] | -0.012 [-0.015, -0.009] | +0.013 [-0.005, +0.032] | -0.066 [-0.130, -0.004] |
| without agent_last_utterance | -0.004 [-0.009, -0.000] | +0.045 [+0.023, +0.067] | +0.000 [-0.005, +0.006] | -0.008 [-0.011, -0.005] | +0.018 [+0.000, +0.037] | -0.047 [-0.101, +0.004] |
| without normalization, scored on written text | +0.134 [+0.128, +0.141] | +0.317 [+0.285, +0.350] | +0.004 [-0.002, +0.010] | -0.014 [-0.017, -0.010] | +0.015 [-0.004, +0.034] | +0.000 [-0.045, +0.046] |
| without normalization, scored on ASR-style text | -0.443 [-0.451, -0.436] | +0.325 [+0.293, +0.359] | -0.003 [-0.009, +0.002] | -0.015 [-0.019, -0.012] | +0.017 [-0.001, +0.036] | +0.006 [-0.039, +0.052] |
<!-- table-d-paired:end -->

**What the ablations show.** Each arm is one training run. The paired intervals cover sampling of
test rows, not training variance, and two runs of the same recipe (this full arm and r2) differ by
up to 0.023 macro-F1 and 0.050 on D1's hard subset.
- **Normalization is necessary.** Trained on written text, the model reads punctuation. Scored on
  written text, D1 reaches 0.959 macro-F1. Scored on ASR-style text, which is what a live
  pipeline delivers, it collapses to 0.381, the majority class's level, with ECE 0.394: it calls
  almost every turn incomplete. That is the leak normalization prevents, and the one result far
  outside run-to-run variance.
- **Context adds no measurable headline accuracy on D1–D3.** Without the conversation history or
  the agent's last utterance, D1–D3 macro-F1 moves by 0.018 or less.
- **D4 may use history.** D4 drops 0.066 [0.004, 0.130] without history (n = 200), which is weak
  evidence.
- **The hard subsets cannot be read.** They move in opposite directions (D1 up about 0.04, D2
  down about 0.01), by amounts the full arm's own draw explains as well as the ablation does.

### Operating points

Thresholds chosen on calib: the smallest θ whose calib rate of the costly error stays at or under
5% (false stops for θ_interrupt, premature responses for θ_yield). The test columns show what that
θ does. θ_escalate (in `floorcall_calibration.json`) and θ_oos (in the repository's
`results/thresholds/route_oos.json`) maximise calib macro-F1.

<!-- operating-points:start -->
| Model | θ_interrupt | false stops (test) | missed interruptions (test) | θ_yield | premature responses (test) | added delay, ms (test) |
|---|---|---|---|---|---|---|
| stock Laya | 0.340 | 0.040 | 0.931 | 0.630 | 0.055 | 1599 |
| fine-tuned + temperature | 0.090 | 0.050 | 0.003 | 0.785 | 0.043 | 903 |
<!-- operating-points:end -->

## Known failure modes

- **Latency: every p99 misses its target.** No GPU path meets p99 ≤ 50 ms on the measured laptop
  GPU (the best is 59.0 ms, with CUDA graphs and the event's questions batched in one call), and
  every CPU path misses p99 ≤ 100 ms by 15–22×. With an event's questions in one call, GPU p50
  is 29–56 ms. On a CPU-only host this checkpoint
  is too slow to sit inline.
- **D2: a lexical rule comes close.** A rule on backchannel words and length scores 0.908 macro-F1
  against this checkpoint's 0.937. The paired gap, +0.029 [+0.022, +0.035], is real but small. On
  D2's hard subset the rule scores 0.985 against 0.977. Most of D2 is "uh-huh"; the model's margin
  is where the words alone do not decide.
- **D4 has a low label ceiling.** Its test labels are one person's judgement. When that person
  relabelled 300 messages blind under a revised guideline, the new labels agreed with their first
  pass at accuracy 0.783, Cohen's κ 0.551. The training labels came from an LLM (`gpt-oss-120b`). On calib it agreed with the
  test labeller at κ 0.420 and under-escalated, with escalate recall 0.426. Read against that, D4's
  macro-F1 of 0.681 at θ_escalate ties TF-IDF + logistic regression (paired difference
  0.000 [−0.061, +0.060], n = 200). Calibration is where the model differs: ECE 0.064 against
  0.167.
- **D1 is text only.** It cannot hear falling intonation, and SwDA is 1990–91 telephone
  conversation between two people, not users talking to an agent. Truncations that end on a
  turn-final word (D1's hard subset) are right 72.5% of the time.
- **ASR noise costs accuracy** on every decision (Table C). At noise level 0.20, D2 drops from
  0.937 to 0.728 macro-F1.
- **Four misses on the scripted replay demos.** The repository replays eight scripted banking calls
  through a naive agent and through this model. These are illustrative demos, not an evaluation
  set, and the scripts were frozen before the model first ran on them. The model acted as the
  script wanted at 14 of 18 points. The four misses:
  1. *"I never bought anything there"* was routed to `transactions`, not `report_fraud`.
  2. Side talk, *"honey can you grab the door"*, stopped the agent: after "honey can",
     p(interruption) was 0.73, and the wanted action was to ignore it.
  3. *"this is ridiculous, I've explained this three times now,"* drew a reply mid-thought
     (p(turn_complete) 0.88), cutting the user off.
  4. *"no, stop, put me through to a human now"* stopped the agent correctly, but the decision
     fires 600 ms into the speech. After "no, stop," p(escalate) was 0.48, under θ_escalate 0.505,
     so there was no hand-off until the next pause.

## Training data

| Decision | Source | Licence | Train | Calib | Test |
|---|---|---|---|---|---|
| D1 `turn_complete` | Switchboard Dialog Act Corpus (SwDA), Potts' distribution | CC BY-NC-SA 3.0 | 49,381 | 8,201 | 14,998 |
| D2 `barge_in` | SwDA | CC BY-NC-SA 3.0 | 44,187 | 7,407 | 13,360 |
| D3 `route` | CLINC150, banking domain + out-of-scope | CC BY 3.0 | 1,750 | 400 | 1,449 |
| D4 `escalate` | Customer Support on Twitter (Thought Vector), banking threads | CC BY-NC-SA 4.0 | 6,000 | 100 | 200 |

- **SwDA** extends LDC's Switchboard-1 Release 2 (LDC97S62) with dialog-act annotations made at
  UC Boulder (Jurafsky, Shriberg and Biasca, 1997). Christopher Potts' CSV distribution is licensed
  CC BY-NC-SA 3.0 Unported. Splits are by conversation (70/10/20), shared by D1 and D2, so no test
  conversation is seen through either task.
- **CLINC150** (Larson et al., 2019), from `clinc/oos-eval`.
- **Customer Support on Twitter** (Thought Vector, Kaggle). Banking and payments brands only, one
  customer message per thread, split by thread. D4's training labels were produced by
  `gpt-oss-120b` (Apache-2.0) from the test labeller's written guidelines; its calib and test
  labels are a person's, never an LLM's.
- All text was lowercased with punctuation removed before training, as at inference.

## Licence

**The weights are released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).**
Two of the training sources are non-commercial and share-alike: SwDA (CC BY-NC-SA 3.0) and
Customer Support on Twitter (CC BY-NC-SA 4.0). Whether model weights are an adaptation of their
training data is legally unsettled, so this release takes the conservative reading and carries
those terms forward:

- **Non-commercial:** no commercial use of these weights or of models derived from them. Thought
  Vector, the Twitter corpus's provider, asks commercial users of that data to contact it.
- **Share-alike:** a model derived from these weights, for example by further fine-tuning, must be
  released under the same licence.
- **Attribution:** credit floorcall, and the sources below.

It builds on:

- **Laya**, © Convai Innovations, Apache License 2.0; its text is in
  `LICENSE-laya-Apache-2.0.txt`. These weights are modified from Laya's, revision `55cf4c4`.
- **ModernBERT-large**, Answer.AI, Apache License 2.0: Laya's encoder, and the tokenizer here.
- **SwDA**: Jurafsky, Shriberg and Biasca (1997), *Switchboard SWBD-DAMSL Shallow-Discourse-Function
  Annotation Coders Manual*; Christopher Potts' distribution, CC BY-NC-SA 3.0.
- **CLINC150**: Larson et al. (2019), *An Evaluation Dataset for Intent Classification and
  Out-of-Scope Prediction*, EMNLP-IJCNLP; CC BY 3.0.
- **Customer Support on Twitter**: Thought Vector, Kaggle; CC BY-NC-SA 4.0.

This licence covers the weights and this card. floorcall's code is Apache-2.0, in its own
repository.
