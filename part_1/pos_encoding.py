"""1.1 Positional encodings (absolute learned + sinusoidal)."""
import math
import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor):
        # x: (B, T, d_model) — we only need its T and device
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device)
        pos_emb = self.emb(pos)  # (T, d_model)
        return x + pos_emb.unsqueeze(0)  # broadcast over batch

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        return x + self.pe[:T].unsqueeze(0)