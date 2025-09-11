# grpo_loss.py
from __future__ import annotations
import torch
from dataclasses import dataclass

@dataclass
class PolicyOnlyLossOut:
    policy_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    kl_ref: torch.Tensor
    total_loss: torch.Tensor


def ppo_policy_only_losses(new_logp, old_logp, adv, clip_ratio=0.2, ent_coef=0.0,
                           kl_coef: float = 0.0, kl_mean: torch.Tensor | None = None):
    """
    PPO-style clipped policy loss, *policy only* (no value head),
    plus a separate KL(π||π_ref) penalty term:  total = L_PPO + kl_coef * KL.
    Inputs are flat over action tokens: new_logp, old_logp, adv: (N_act,).
    kl_mean is a scalar tensor (mean over action tokens).
    """
    device = new_logp.device if new_logp.is_cuda else None
    if new_logp.numel() == 0:
        zero = torch.tensor(0.0, device=device)
        return PolicyOnlyLossOut(zero, zero, zero, zero, zero)

    ratio = torch.exp(new_logp - old_logp)  # (N,)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    policy_loss = -torch.mean(torch.min(unclipped, clipped))

    entropy = -new_logp.mean() if ent_coef != 0.0 else new_logp.new_tensor(0.0)
    approx_kl = torch.mean(old_logp - new_logp)

    kl_ref = kl_mean if kl_mean is not None else new_logp.new_tensor(0.0)

    total = policy_loss - ent_coef * entropy + kl_coef * kl_ref # entropy bonus was not used in original GRPO paper
    return PolicyOnlyLossOut(policy_loss, entropy, approx_kl, kl_ref, total)
