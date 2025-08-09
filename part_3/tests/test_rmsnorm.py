import torch
from rmsnorm import RMSNorm

def test_rmsnorm_shapes():
    x = torch.randn(2,3,8)
    rn = RMSNorm(8)
    y = rn(x)
    assert y.shape == x.shape