# Runbook: Table D ablations on Kaggle

Table D (CLAUDE.md §12) trains the model three more times, each with part of the input taken away:
no ASR-style normalization, no `recent_turns`, no `agent_last_utterance`. The laptop cannot take
more multi-hour training runs (DECISIONS.md D-037: it crashed from heat), so these run on Kaggle,
started by Yash. Nothing here runs locally except building the bundle.

## Decided (DECISIONS.md D-038)

**1. A `full` arm on Kaggle is Table D's reference.** Each ablation row is read against a
reference row from the same recipe. Kaggle's GPUs (T4, P100) have no fast bf16, so arms there
train in fp16 with loss scaling on different hardware. The local r2 trained in bf16 on the RTX 5070
Ti. Comparing Kaggle ablations with r2 would mix the ablation's effect with precision and hardware.
With a Kaggle `full` arm, all four Table D rows share the same conditions. Table D then compares
Kaggle arms with each other only; Table A's fine-tuned row stays r2.

**2. 2 epochs for every arm**, full and ablations alike, because r2's dev loss picked epoch 2 of
4. The best checkpoint is still the epoch with the lowest dev loss. The notebook's `EPOCHS` is 2;
leave it. The learning-rate schedule spans the run's own 2 epochs. A T4 is several times slower
than the laptop, and Kaggle stops a session at 12 hours, with about 30 GPU hours a week. The smoke
run's time estimate is for information only.

## 1. Build the dataset (local, CPU only)

From a clean, pushed tree:

```bash
uv run python scripts/kaggle_bundle.py
```

This writes `dist/kaggle/floorcall-table-d/`: the source at HEAD (`git archive`), the processed
train and calib files of all four decisions, and the frozen test sets. It also writes
`floorcall/BUNDLE.json`, with the commit, the pinned package versions and a sha256 per data file.
There are no raw corpora, no label files and no keys. The script refuses a dirty tree, because
every Kaggle result cites the bundle's commit as its code.

The dataset id in `dataset-metadata.json` is `yashraz/floorcall-table-d`, built from two constants
in `scripts/kaggle_bundle.py` (`KAGGLE_OWNER`, `DATASET_SLUG`). An "Invalid Owner Id" from the CLI
means `KAGGLE_OWNER` is not the Kaggle username of the API token in use.

## 2. Upload it (private)

```bash
pip install kaggle                  # once; API token in ~/.kaggle/kaggle.json (Kaggle → Settings → API)
kaggle datasets create -p dist/kaggle/floorcall-table-d --dir-mode zip
```

**The dataset must be private. Never pass `--public`.** A new dataset is private unless it is
given. It holds the frozen test sets and is derived from Customer Support on Twitter
(CC BY-NC-SA 4.0), SwDA (CC BY-NC-SA 3.0) and CLINC150 (CC BY 3.0). After uploading:

1. Open the URL the CLI prints. The dataset page must show **Private**, and it must not appear in a
   logged-out browser.
2. **Settings → Sharing** must list no one but you.

The bundle script has already scanned every file (D-038): names like `.env` or `kaggle.json`, key
patterns, and every value from the local `.env`, inside the gzipped data too. It refuses the bundle
on any hit. After a later commit, rebuild and push a new version with
`kaggle datasets version -p dist/kaggle/floorcall-table-d -m "<commit>" --dir-mode zip`. A new
version keeps the dataset private.

## 3. The notebook

1. Kaggle → **Create → New notebook → File → Import notebook**, then choose
   `kaggle/floorcall_table_d.ipynb`.
2. **Add input**, then pick the `floorcall-table-d` dataset. It mounts at
   `/kaggle/input/floorcall-table-d/`; if the path differs, set `BUNDLE` in the first cell.
3. Session options: **Accelerator: GPU T4 x2** (one GPU is used) or P100, and **Internet: on**,
   for pip and the pinned Laya checkpoint.
4. In the first cell set `ARM`. `EPOCHS` stays 2 (D-038). Optionally set `SMOKE_ONLY = True`
   and run it interactively once: it installs, verifies the bundle, runs the smoke test and prints
   the time estimate in minutes.
5. For the real run: **Save Version → Save & Run All (Commit)**. It runs headless for up to 12 hours
   and keeps `/kaggle/working` as the version's output.

What the notebook does, and where it stops if anything is off:

| Cell | Does | Stops if |
|---|---|---|
| bundle | copies the bundle to `/kaggle/working/floorcall`, re-hashes every data file | any file differs from `BUNDLE.json` |
| install | pinned versions from `BUNDLE.json`; Kaggle's own CUDA torch is kept | pip fails |
| environment | prints torch, GPU and capability; on capability < 8, sets `FLOORCALL_TRAIN__AMP_DTYPE=fp16` | — |
| smoke | the whole training path on 600 train rows per task, 2 epochs of 64 rows; prints the time estimate | anything fails: the fp16 path has not run on a T4 before this cell |
| arm | `floorcall train run --ablation ARM`, then `floorcall eval ablation` | any failure |
| outputs | copies `run.json`, the train log, the calibration, the Table D JSON and the per-row predictions to `outputs/ARM.zip` | — |

## 4. Bring the results home

1. From the finished version's **Output** tab, download `outputs/<ARM>.zip`.
2. Unzip it. Copy its `*.json` Table D rows into `results/table_d/`; the files are named
   `<decision>.<variant>.json`. Keep `run.json`, `train_log.jsonl` and
   `floorcall_calibration.json` under `results/training/kaggle_<ARM>/` for the record.
3. Check each row's `code` is `<bundle commit>-kaggle`, and that its `test_sha256` matches
   `data/test_frozen/MANIFEST.json`.
4. `uv run floorcall eval readme`, run the tests, then commit.

Once the four arms are in, Table D is complete. Log in DECISIONS.md which arms ran, their epochs,
the GPU type, and the reference arm the ablations are read against.
