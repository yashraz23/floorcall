# Labelling guidelines: D4 escalate (v1)

These guidelines are versioned with the labels: every label records `"guidelines": "v1"`. If they
change, the version changes, and labels made under different versions are never mixed in one test
set.

## The question

The model is asked **"Should this conversation be handed to a human?"** about one customer message.
The customer plays the user. The company's support account plays the agent. You see the agent's
last reply, if there was one, and up to two earlier turns.

## Label `y` (escalate) when either holds

1. **The customer asks for a person**: a human, a representative, a supervisor or manager, a phone
   call, "someone who can actually help".
2. **The customer is frustrated in a way support is not resolving**: repeated contact ("I already
   DMed you", "third time"), "nobody is helping", hostility toward the support itself, or a threat
   to close the account, leave, or go to a regulator or the press.

## Label `n` when

- It is a routine question or request, even about a problem ("my card was declined, why?").
- The customer is supplying requested information, thanking, confirming ("ok, done"), or praising.
- There is annoyance at a product or situation, but no sign that support has failed them ("ugh, the
  app is down").

## Skip (`s`) when

You cannot tell: sarcasm you cannot read, context too thin to judge, not English, spam, or
unreadable. Skipped messages never enter the test set. Skip freely; a wrong label is worse than a
missing one.

## Tie-breaker

*If you were the team lead watching this conversation live, would you want a person to take over
now?*

## What you will not see

Candidates were drawn in four equal strata: messages with and without common escalation words,
crossed with openers and replies to an agent. The tool never shows which stratum a message came
from, or any model prediction, so neither can steer a label.
