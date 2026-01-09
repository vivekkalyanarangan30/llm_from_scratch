import torch.nn as nn

class SwiGLU(nn.Module):
    """SwiGLU FFN: (xW1) ⊗ swish(xW2) W3  with expansion factor `mult`.
    """
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = mult * dim
        self.w1 = nn.Linear(dim, inner, bias=False) # gate
        self.w2 = nn.Linear(dim, inner, bias=False) # activation
        self.w3 = nn.Linear(inner, dim, bias=False) # projection
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        a = self.w1(x)  # gate
        b = self.act(self.w2(x))  # activation
        return self.drop(self.w3(a * b))  # projection