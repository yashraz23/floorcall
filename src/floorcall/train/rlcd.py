"""The RLCD training objective, reproduced from Laya's fine-tuning notebook.

RLCD trains the decision head as a policy whose "action" is a reported probability distribution,
rewarded by strictly proper scoring rules. A proper scoring rule is maximised in expectation only
by reporting the true distribution, so the reward pushes towards calibrated probabilities rather
than confident ones. Per row:

1. Take the model's logits z. Sample G perturbed copies z + eps, with eps ~ N(0, sigma^2) projected
   to zero mean over the row's real options (a shift that changes nothing is not exploration).
2. Score each perturbed distribution q = softmax(z + eps) against the gold distribution with
   log score + w_sph * spherical score (minus w_rps * RPS on ordinal `score` questions).
3. Advantage = reward minus the group mean, normalised by its std. That is the GRPO baseline:
   no value network, the group is its own baseline.
4. Gaussian policy log-likelihood log p(z + eps | z) = -|eps|^2 / (2 sigma^2), up to a constant.
   Its gradient moves z towards perturbations that scored above the group mean.
5. Add soft cross-entropy against the gold distribution, weight 1.0, as the notebook does. On
   its own the RL term is a noisy gradient estimate; the CE term is the stable signal it refines.

floorcall trains with the RL term's weight at 0 (`TrainSettings.rl_weight`, DECISIONS.md D-035
amendment 1). Normalising advantages to unit size keeps the term's gradient full-size however
small the reward differences get, and that gradient grows as 1/sigma. Measured before clipping on
the stock weights, it was 2-15x the cross-entropy gradient at sigma 0.4 and 11-45x at 0.1
(results/training/r1_stopped). Once cross-entropy had saturated, it kept driving logits apart, and
the first full run ended near-certain on D1 (mean confidence 0.995) with its wrong answers.

Everything here is plain torch. The reward function is Laya's own, reached through the adapter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

# (q, target, qtype, mask) -> reward per (group, row). Bound to LayaDecider.proper_reward with the
# configured weights, so this module never imports laya.
RewardFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class StepLoss:
    loss: torch.Tensor  # what backward() is called on
    rl: float
    ce: float
    reward: float


def sigma_for_epoch(epoch: int, epochs: int, start: float, end: float) -> float:
    """Exploration noise, annealed linearly from `start` (first epoch) to `end` (last)."""
    progress = epoch / max(1, epochs - 1)
    return start + (end - start) * progress


def rlcd_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    marker_mask: torch.Tensor,
    qtype: torch.Tensor,
    *,
    sigma: float,
    group_size: int,
    reward_fn: RewardFn,
    ce_weight: float,
    rl_weight: float = 1.0,
    generator: torch.Generator | None = None,
) -> StepLoss:
    """The notebook's per-micro-batch loss, before division by gradient accumulation.

    logits: (N, K) decision-head scores, padded options at -1e4.
    target: (N, K) gold distribution per row, zero on padded options.
    marker_mask: (N, K) True on real options.
    """
    logits = logits.float()
    mask = marker_mask
    k = mask.sum(-1, keepdim=True).float()

    eps = (
        torch.randn((group_size, *logits.shape), device=logits.device, generator=generator)
        * sigma
        * mask
    )
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

    with torch.no_grad():
        r = reward_fn(q, target.unsqueeze(0), qtype, mask)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)

    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
    loss_rl = -(adv * logp).mean()
    loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
    # rl_weight 0 leaves the policy-gradient term out of the graph entirely (D-035 amendment 1);
    # its value and the reward are still reported.
    loss = ce_weight * loss_ce + (rl_weight * loss_rl if rl_weight else 0.0)
    return StepLoss(
        loss=loss,
        rl=float(loss_rl.detach()),
        ce=float(loss_ce.detach()),
        reward=float(r.mean()),
    )
