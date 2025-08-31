from __future__ import annotations
import torch, torch.nn as nn

class TopKGate(nn.Module):
    """Top‑k softmax gating with Switch‑style load‑balancing aux loss.
    Args:
      dim: input hidden size
      n_expert: number of experts
      k: number of experts to route per token (1 or 2 typical)
    Returns:
      (indices, weights, aux_loss) where
        indices: (S, k) long, expert ids for each token
        weights: (S, k) float, gate weights (sum ≤ 1 per token)
        aux_loss: scalar load‑balancing penalty
    """
    def __init__(self, dim: int, n_expert: int, k: int = 1):
        super().__init__()
        assert k >= 1 and k <= n_expert
        self.n_expert = n_expert
        self.k = k
        self.w_g = nn.Linear(dim, n_expert, bias=True)

    def forward(self, x: torch.Tensor):
        # x: (S, C) where S = tokens (batch * seq)
        logits = self.w_g(x)                  # (S, E)
        probs = torch.softmax(logits, dim=-1) # (S, E)
        topk_vals, topk_idx = torch.topk(probs, k=self.k, dim=-1)  # (S,k)

        # Load‑balancing aux loss (Switch):
        S, E = probs.size(0), probs.size(1)
        # importance: avg prob per expert
        importance = probs.mean(dim=0)                 # (E,)
        # load: fraction of tokens assigned as primary (top‑1 hard assignment)
        hard1 = topk_idx[:, 0]                         # (S,)
        load = torch.zeros(E, device=x.device)
        load.scatter_add_(0, hard1, torch.ones_like(hard1, dtype=load.dtype))
        load = load / max(S, 1)
        aux_loss = (E * (importance * load).sum())
        # print("*"*50)
        # print(probs, importance, hard1, load, aux_loss)
        # print("*"*50)

        return topk_idx, topk_vals, aux_loss