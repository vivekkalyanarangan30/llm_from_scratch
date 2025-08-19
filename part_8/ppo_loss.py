from __future__ import annotations
import torch, torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class PPOLossOut:
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    total_loss: torch.Tensor


def ppo_losses(new_logp, old_logp, adv, new_values, old_values, returns,
               clip_ratio=0.2, vf_coef=0.5, ent_coef=0.0):
    # policy
    ratio = torch.exp(new_logp - old_logp)  # (N,)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
    policy_loss = -torch.mean(torch.min(unclipped, clipped))

    # value (clip optional → here: simple MSE)
    value_loss = F.mse_loss(new_values, returns)

    # entropy bonus (we approximate entropy via -new_logp mean; strictly needs full dist)
    entropy = -new_logp.mean()

    # approx KL for logging
    approx_kl = torch.mean(old_logp - new_logp)

    total = policy_loss + vf_coef * value_loss - ent_coef * entropy
    return PPOLossOut(policy_loss, value_loss, entropy, approx_kl, total)