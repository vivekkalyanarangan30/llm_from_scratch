import torch
from policy import PolicyWithValue

def test_policy_shapes():
    B,T,V = 2, 16, 256
    pol = PolicyWithValue(vocab_size=V, block_size=T, n_layer=2, n_head=2, n_embd=64)
    x = torch.randint(0, V, (B,T))
    logits, values, loss = pol(x, None)
    assert logits.shape == (B,T,V)
    assert values.shape == (B,T)
