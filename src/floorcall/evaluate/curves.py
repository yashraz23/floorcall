"""The two headline tradeoff curves (CLAUDE.md §6, §12) and how an operating point is chosen.

theta_interrupt: the agent stops talking when p(interruption) >= theta.
  - false stop: a non-interruption (backchannel or noise) that stops the agent. Reported over all
    non-interruptions, and over backchannels alone (the "agent halts for uh-huh" case).
  - missed interruption: an interruption the agent talks over.

theta_yield: the agent responds at a pause when p(turn_complete) >= theta.
  - premature response: an unfinished turn the agent answers, cutting the user off.
  - added delay: a finished turn the model declines to answer waits for the silence safety net.
    The pause event fires after `vad_pause_ms` of silence and the net after `max_wait_ms`, so each
    such turn costs (max_wait_ms - vad_pause_ms), and the mean added delay over finished turns is
    that times the fallback rate. A delay model is needed because text alone has no clock; this one
    is the policy of floorcall.policy, stated so it can be checked.

Operating points are chosen on the calib split, never on test: the smallest theta whose calib rate
of the costly error (false stops; premature responses) is at or under its target. That keeps as
many true stops and prompt responses as the target allows. The test set then reports what that
theta actually does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def interrupt_curve(
    p_interruption: FloatArray,
    y: npt.NDArray[np.integer],
    labels: tuple[str, ...],
    thetas: FloatArray,
) -> dict[str, FloatArray]:
    is_int = y == labels.index("interruption")
    is_bc = y == labels.index("backchannel")
    stop = np.asarray(p_interruption)[None, :] >= np.asarray(thetas)[:, None]
    return {
        "theta": np.asarray(thetas, dtype=np.float64),
        "false_stop": stop[:, ~is_int].mean(axis=1),
        "false_stop_backchannel": stop[:, is_bc].mean(axis=1),
        "missed": (~stop[:, is_int]).mean(axis=1),
    }


def yield_curve(
    p_complete: FloatArray,
    y: npt.NDArray[np.integer],
    thetas: FloatArray,
    *,
    vad_pause_ms: float,
    max_wait_ms: float,
) -> dict[str, FloatArray]:
    complete = y == 1  # option order ("false", "true")
    respond = np.asarray(p_complete)[None, :] >= np.asarray(thetas)[:, None]
    fallback = (~respond[:, complete]).mean(axis=1)
    return {
        "theta": np.asarray(thetas, dtype=np.float64),
        "premature": respond[:, ~complete].mean(axis=1),
        "fallback": fallback,
        "added_delay_ms": fallback * (max_wait_ms - vad_pause_ms),
    }


@dataclass(frozen=True)
class Choice:
    theta: float
    rate: float
    feasible: bool  # False: no theta met the limit; the largest one is returned


def choose_threshold(thetas: FloatArray, rate: FloatArray, *, limit: float) -> Choice:
    """The smallest theta whose rate is at or under `limit`. `rate` falls as theta rises."""
    ok = np.flatnonzero(np.asarray(rate) <= limit)
    i = int(ok[0]) if ok.size else len(thetas) - 1
    return Choice(theta=float(thetas[i]), rate=float(rate[i]), feasible=bool(ok.size))
