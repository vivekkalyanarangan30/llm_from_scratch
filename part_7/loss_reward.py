from __future__ import annotations
import torch, torch.nn.functional as F

def bradley_terry_loss(r_pos: torch.Tensor, r_neg: torch.Tensor) -> torch.Tensor:
    """-log sigma(r_pos - r_neg) = softplus(-(r_pos - r_neg))
    https://docs.pytorch.org/docs/stable/generated/torch.nn.Softplus.html"""
    diff = r_pos - r_neg
    return F.softplus(-diff).mean()


def margin_ranking_loss(r_pos: torch.Tensor, r_neg: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """https://docs.pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html"""
    y = torch.ones_like(r_pos)
    return F.margin_ranking_loss(r_pos, r_neg, y, margin=margin)