# Labelling guidelines: D4 escalate (v2)

These guidelines are versioned with the labels: every label records the version it was made under.
Labels made under different versions are never mixed in one eval set (DECISIONS.md D-033).

## The question

The model is asked **"Should this conversation be handed to a human?"** about one customer message.
The customer plays the user. The company's support account plays the agent. You see the agent's
last reply, if there was one, and up to two earlier turns.

## Guideline v2

Escalate (y) if any of: (1) Asks for a human, a call, or a manager, or says they can't reach or get help from support. (2) Says they already raised it (called, DM'd, visited, told a rep) and it's still unresolved or past a promised time. (3) Says they'll leave, close the account, switch banks, or take it public/viral.

Not escalate (n): venting or insults with no prior contact and no threat to leave; a first report of a problem, even an angry one; questions, thanks, "sent you a DM"; fraud or phishing reports; general sarcasm about the company.

If unsure: apply the three rules literally. If none clearly fits, n.

## What you will not see

Candidates were drawn in four equal strata: messages with and without common escalation words,
crossed with openers and replies to an agent. The relabel tool (`floorcall label
escalate-relabel`) shows the sample shuffled, test and calib together, with the same context as the
first pass and this guideline above every message. It never shows a message's split or stratum,
your v1 label, or any model or LLM label, so none of them can steer a label.

## Earlier versions

Guideline v1 (2026-09-27) governed the first pass, `data/labels/escalate.labels.jsonl`, which is
kept unchanged. Its text is in git: `git show 808f0ef:docs/labelling-escalate.md`.
