import torch
from block_hybrid import HybridFFN

def test_hybrid_ffn_blend():
    B,T,C = 1, 4, 16
    ffn = HybridFFN(dim=C, alpha=0.3, n_expert=3, k=2)
    x = torch.randn(B,T,C)
    y, aux = ffn(x)
    assert y.shape == x.shape
    assert aux.item() >= 0.0