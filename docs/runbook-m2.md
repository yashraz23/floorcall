# Milestone 2 runbook: from frozen D4 to Tables A–D

Every command below is committed code. Run them in order, from a clean tree, so each results file
records a commit that can reproduce it. Times are for the RTX 5070 Ti Laptop GPU, measured where
marked, projected otherwise.

## 0. Preconditions

```bash
uv run floorcall data freeze-escalate      # after `floorcall label escalate` (test, then --split calib)
uv run floorcall data verify               # all four test sets present and hash-matched
```

`floorcall train run` refuses to start until all four test sets are frozen (D-009, D-023). D4
evaluates on test v2 and calibrates on calib v2 (D-033). Its training rows are the 6,000-message
train pool labelled by prompt v3, `source=llm_v3` (D-035), built by
`uv run floorcall data escalate-llm-label --split train` (cached, so a rerun costs nothing). D4
rows are drawn class-balanced each epoch (`TrainSettings.balance_classes`), and calibration also
picks D4's threshold on calib v2 by macro-F1. Stock Laya's threshold is picked the same way by
`uv run floorcall eval stock-threshold`.

## 1. Train and calibrate the main model (projected ~3 h)

```bash
uv run floorcall train run --out checkpoints/main
```

This builds items from the train splits, packed exactly as served (~10 min), trains 4 epochs of
about 50k rows each (~46 min an epoch at the 18 rows/s measured in the M0 spike), saves in Laya's
layout, reloads the saved checkpoint, and fits one temperature per decision on calib. The outputs
are `run.json` (settings, data hashes, test manifest, code commit, history), `train_log.jsonl` and
`floorcall_calibration.json`. A temperature flagged "at a bound" needs a look before anything is
reported.

## 2. Score it

```bash
uv run floorcall eval checkpoint --checkpoint checkpoints/main    # Table A: fine-tuned, + temperature
uv run floorcall eval curves --checkpoint checkpoints/main        # operating points, reliability
uv run floorcall eval robustness --checkpoint checkpoints/main    # Table C
```

Check that Table C's noise-0.00 row equals Table A's "+ temperature" row (D-024).

## 3. Ablations for Table D (projected ~3 h each)

```bash
uv run floorcall train run --ablation no_history   --out checkpoints/no_history
uv run floorcall train run --ablation no_agent     --out checkpoints/no_agent
uv run floorcall train run --ablation no_normalize --out checkpoints/no_normalize
uv run floorcall eval ablation --checkpoint checkpoints/main          # the "full" reference row
uv run floorcall eval ablation --checkpoint checkpoints/no_history
uv run floorcall eval ablation --checkpoint checkpoints/no_agent
uv run floorcall eval ablation --checkpoint checkpoints/no_normalize  # scored on written and ASR text
```

## 4. Latency on the trained checkpoint, figures, README

```bash
FLOORCALL_LAYA__CHECKPOINT=checkpoints/main FLOORCALL_LAYA__REVISION= uv run floorcall eval latency
uv run floorcall eval figures
uv run floorcall eval readme
uv run pytest                              # includes the README-drift check
```

Latency depends on architecture and input length, not on weights. This rerun confirms it, and it
also puts the release checkpoint in each Table B row's provenance.
