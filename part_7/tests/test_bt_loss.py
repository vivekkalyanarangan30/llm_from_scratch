import torch
from loss_reward import bradley_terry_loss

def test_bradley_terry_monotonic():
    pos = torch.tensor([2.0, 3.0])
    neg = torch.tensor([1.0, 1.5])
    l1 = bradley_terry_loss(pos, neg)
    l2 = bradley_terry_loss(pos+1.0, neg)  # increase margin
    assert l2 < l1