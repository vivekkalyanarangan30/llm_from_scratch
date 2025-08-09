import torch
from rope import RoPECache, apply_rope

def test_rope_rotation_shapes():
    B,H,T,D = 1, 2, 5, 8
    rc = RoPECache(head_dim=D, max_pos=32)
    q = torch.randn(B,H,T,D)
    k = torch.randn(B,H,T,D)
    pos = torch.arange(0, T)
    cos, sin = rc.get(pos)
    q2, k2 = apply_rope(q, k, cos, sin)
    assert q2.shape == q.shape and k2.shape == k.shape