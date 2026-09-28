# Datasets

Built by `uv run floorcall data build` from sources pinned by SHA256 (`floorcall.data.download`).
Train and calib go to `data/processed/` (gitignored, rebuilt freely). Test sets are frozen in
`data/test_frozen/`, hashed in `MANIFEST.sha256`, committed, and never edited
(`floorcall.data.freeze`). Class balance for every split is in
`data/processed/cards/<decision>.json`. Construction rules and their reasons are in
`docs/DECISIONS.md` D-015 to D-019.

| Decision | Source | Licence | Train | Calib | Test | Test balance |
|---|---|---|---|---|---|---|
| D1 turn_complete | SwDA | CC BY-NC-SA 3.0 | 49,381 | 8,201 | 14,998 | false 8,843 / true 6,155 |
| D2 barge_in | SwDA | CC BY-NC-SA 3.0 | 44,187 | 7,407 | 13,360 | backchannel 5,659 / interruption 6,700 / noise 1,001 |
| D3 route | CLINC150 banking + oos | CC BY 3.0 | 1,750 | 400 | 1,449 | 29–30 per intent, out_of_scope 1,000 |
| D4 escalate | Customer Support on Twitter, banking threads, hand-labelled by Yash | CC BY-NC-SA 4.0 | TODO | TODO (from 200 candidates) | TODO (≥300 of 400 candidates) | TODO |

**Splits.** SwDA is split by conversation (sha256 of seed and conversation id: 70/10/20). D1 and D2
share the mapping, so multi-task training cannot see a test conversation through either task.
CLINC uses its own splits, with test and calib rows that duplicate a train query removed.

**Hard subsets** (test): D1 775 truncations ending on a turn-final word; D2 5,012 rows built from
a surface form both classes use ("yeah" vs "yeah but that's not what i asked"). D3 has none.

**Packing** (`scripts/pack_stats.py`, real tokenizer, `results/pack_stats.json`). Every test row fits
its event's budget. History is trimmed for 64% of D1 and 43% of D2 states. The user's own words are
trimmed for 0.4% of D1 states (monologues over ~300 tokens) and never for D2 or D3.

**Known gaps.** SwDA is two humans on the phone in 1990–91, not a user talking to an agent. There
is no audio, so no prosody: a text model cannot hear a falling intonation. Both are stated in the
README next to the results.

**Licences.** SwDA is CC BY-NC-SA 3.0, so any dataset derived from it is non-commercial and
share-alike. That suits a portfolio release, and the dataset cards will say so.
