from __future__ import annotations
import torch, torch.nn as nn
import sys
from pathlib import Path as _P
# Try user’s structure first
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))
try:
    from model_utils.model_modern import GPTModern  # user-custom path
except Exception:
    from model_modern import GPTModern  # fallback

class PolicyWithValue(nn.Module):
    """Policy network = SFT LM + tiny value head.
    NOTE: For simplicity we place value head on top of LM logits (vocab→1).
    This avoids depending on hidden-state internals while keeping the tutorial runnable.
    """
    def __init__(self, vocab_size: int, block_size: int, n_layer=4, n_head=4, n_embd=256,
                 use_rmsnorm=True, use_swiglu=True, rope=True, dropout=0.0):
        super().__init__()
        self.lm = GPTModern(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer,
                            n_head=n_head, n_embd=n_embd, use_rmsnorm=use_rmsnorm,
                            use_swiglu=use_swiglu, rope=rope, dropout=dropout)
        # value head over logits (toy). Shapes: (B,T,V) -> (B,T,1) -> (B,T)
        self.val_head = nn.Linear(vocab_size, 1, bias=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None):
        # Delegate LM forward; returns logits (B,T,V), loss, _
        logits, loss, _ = self.lm(x, y)
        values = self.val_head(logits).squeeze(-1)  # (B,T)
        return logits, values, loss

    def generate(self, *args, **kwargs):
        return self.lm.generate(*args, **kwargs)