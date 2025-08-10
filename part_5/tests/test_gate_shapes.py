import torch
from gating import TopKGate

def test_gate_topk_shapes():
    S, C, E, K = 32, 64, 4, 2
    x = torch.randn(S, C)
    gate = TopKGate(C, E, k=K)
    idx, w, aux = gate(x)
    assert idx.shape == (S, K)
    assert w.shape == (S, K)
    assert aux.ndim == 0
    # per-token weights are non-negative and <=1
    assert torch.all(w >= 0)
    assert torch.all(w <= 1)