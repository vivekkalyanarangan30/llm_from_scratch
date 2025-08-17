import torch
from model_reward import RewardModel

def test_reward_shapes_and_grad():
    B,T,V = 4, 16, 256
    m = RewardModel(vocab_size=V, block_size=T, n_layer=2, n_head=2, n_embd=64)
    x = torch.randint(0, V, (B,T))
    r = m(x)
    assert r.shape == (B,)
    # gradient flows
    loss = (r**2).mean()
    loss.backward()
    grads = [p.grad is not None for p in m.parameters()]
    assert any(grads)