# Data licences

floorcall's code is Apache-2.0. **That licence does not cover this directory.** Each file here
is derived from a public corpus and carries that corpus's licence, as below. Use of these files
must follow it, including its attribution, non-commercial and share-alike terms.

| Files | Derived from | Licence | Attribution |
|---|---|---|---|
| `test_frozen/turn_complete.test.v1.jsonl.gz`, `test_frozen/barge_in.test.v1.jsonl.gz`, `processed/cards/turn_complete.json`, `processed/cards/barge_in.json` | Switchboard Dialog Act Corpus (SwDA), Christopher Potts' distribution | [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) | Jurafsky, Shriberg and Biasca (1997); SwDA extends LDC's Switchboard-1 Release 2 (LDC97S62) |
| `test_frozen/route.test.v1.jsonl.gz`, `processed/cards/route.json` | CLINC150 (`clinc/oos-eval`) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | Larson et al. (2019), *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction* |
| `labels/*` (by message id: hand labels, LLM labels, the relabel sample, `PRIVATE.sha256`), `processed/cards/escalate.json` | Customer Support on Twitter (Thought Vector, Kaggle) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) | Thought Vector, *Customer Support on Twitter* |

`test_frozen/MANIFEST.json` and `MANIFEST.sha256` record each test set's source, licence and hash.

## D4: no Twitter text here

The customer messages behind D4 (`escalate`) are real people's tweets. Some carry their phone
numbers, names and reference numbers, so this repository does not publish them (docs/DECISIONS.md
D-049). It publishes what D4 needs besides the text:

- the labels by message id;
- the seeds;
- the sha256 of each of the four files that hold the text:
  - `labels/escalate.candidates.v1.jsonl`
  - `labels/escalate.train_sample.v2.jsonl`
  - `test_frozen/escalate.test.v1.jsonl.gz`
  - `test_frozen/escalate.test.v2.jsonl.gz`

To rebuild them:

```bash
uv run floorcall data restore-escalate
```

This downloads the corpus (pinned by its content hash) and rebuilds the four files byte for byte.
It checks each against its committed sha256 before writing it. The rebuilt files carry the
corpus's licence, CC BY-NC-SA 4.0. They are gitignored, so they are never committed back.
