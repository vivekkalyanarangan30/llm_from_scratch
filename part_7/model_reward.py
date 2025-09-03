from __future__ import annotations
import torch, torch.nn as nn

class RewardModel(nn.Module):
    """Transformer encoder → pooled representation → scalar reward.
    Bidirectional encoder is fine for reward modeling (not used for generation).
    """
    def __init__(self, vocab_size: int, block_size: int, n_layer: int = 4, n_head: int = 4, n_embd: int = 256, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        enc_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward=4*n_embd,
                                               dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer)
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, 1)

    def forward(self, x: torch.Tensor):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)
        pad_mask = (x == 2)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = self.ln(h)
        # masked mean pool over tokens (ignoring pads)
        mask = (~pad_mask).float().unsqueeze(-1)
        h_sum = (h * mask).sum(dim=1)
        len_ = mask.sum(dim=1).clamp_min(1.0)
        pooled = h_sum / len_
        r = self.head(pooled).squeeze(-1)  # (B,)
        return r