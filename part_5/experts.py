from __future__ import annotations
import torch.nn as nn

class ExpertMLP(nn.Module):
    """Single expert MLP (SwiGLU or GELU)."""
    def __init__(self, dim: int, mult: int = 4, swiglu: bool = True, dropout: float = 0.0):
        super().__init__()
        inner = mult * dim
        if swiglu:
            self.inp1 = nn.Linear(dim, inner, bias=False)
            self.inp2 = nn.Linear(dim, inner, bias=False)
            self.act = nn.SiLU()
            self.out = nn.Linear(inner, dim, bias=False)
            self.drop = nn.Dropout(dropout)
            self.swiglu = True
        else:
            self.ff = nn.Sequential(
                nn.Linear(dim, inner), nn.GELU(), nn.Linear(inner, dim), nn.Dropout(dropout)
            )
            self.swiglu = False
    def forward(self, x):
        if self.swiglu:
            a = self.inp1(x); b = self.act(self.inp2(x))
            return self.drop(self.out(a * b))
        else:
            return self.ff(x)