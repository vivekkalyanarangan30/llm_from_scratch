import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from attn_mask import causal_mask

class SingleHeadSelfAttention(nn.Module):
    """1.3 Single-head attention (explicit shapes)."""
    def __init__(self, d_model: int, d_k: int, dropout: float = 0.0, trace_shapes: bool = False):
        super().__init__()
        self.q = nn.Linear(d_model, d_k, bias=False)
        self.k = nn.Linear(d_model, d_k, bias=False)
        self.v = nn.Linear(d_model, d_k, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.trace_shapes = trace_shapes

    def forward(self, x: torch.Tensor):  # x: (B, T, d_model)
        B, T, _ = x.shape
        q = self.q(x)  # (B,T,d_k)
        k = self.k(x)  # (B,T,d_k)
        v = self.v(x)  # (B,T,d_k)
        if self.trace_shapes:
            print(f"q {q.shape}  k {k.shape}  v {v.shape}")
        scale = 1.0 / math.sqrt(q.size(-1))
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,T,T)
        mask = causal_mask(T, device=x.device)
        attn = attn.masked_fill(mask.squeeze(1), float('-inf'))
        w = F.softmax(attn, dim=-1)
        w = self.dropout(w)
        out = torch.matmul(w, v)  # (B,T,d_k)
        if self.trace_shapes:
            print(f"weights {w.shape}  out {out.shape}")
        return out, w