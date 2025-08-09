import torch
from kv_cache import RollingKV

def test_rolling_kv_keep_window_with_sink():
    B,H,D = 1,2,4
    kv = RollingKV(window=4, sink=2)
    for _ in range(10):
        k_new = torch.randn(B,H,1,D)
        v_new = torch.randn(B,H,1,D)
        k,v = kv.step(k_new, v_new)
        # Should never exceed sink+window length
        assert k.size(2) <= 6