from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from rope import RoPECache, apply_rope
from kv_cache import KVCache

class CausalSelfAttentionModern(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0,
                 rope: bool = True, max_pos: int = 4096,
                 sliding_window: int | None = None, attention_sink: int = 0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.d_head = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.use_rope = rope
        self.rope_cache: RoPECache | None = None
        self.max_pos = max_pos
        self.sliding_window = sliding_window
        self.attention_sink = attention_sink

    def _maybe_init_rope(self, device):
        if self.use_rope and self.rope_cache is None:
            self.rope_cache = RoPECache(self.d_head, self.max_pos, device=device)

    def forward(self, x: torch.Tensor, kv_cache: KVCache | None = None, start_pos: int = 0):
        """x: (B,T,C). If kv_cache given, we assume generation (T is small, often 1)."""
        B,T,C = x.shape
        self._maybe_init_rope(x.device)

        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head)
        q, k, v = qkv.unbind(dim=2)  # (B,T,H,D)
        q = q.transpose(1, 2)        # (B,H,T,D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE to *current* q,k; cached k already rotated in previous steps
        if self.use_rope:
            pos = torch.arange(start_pos, start_pos + T, device=x.device)
            cos, sin = self.rope_cache.get(pos)
            q, k = apply_rope(q, k, cos, sin)

        # Concatenate past cache if provided
        if kv_cache is not None:
            k_all = torch.cat([kv_cache.k, k], dim=2)
            v_all = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_all, v_all = k, v

        # Sliding-window + attention-sink (crop keys/values for attention computation)
        if self.sliding_window is not None and k_all.size(2) > (self.sliding_window + self.attention_sink):
            sink = self.attention_sink
            k_all = torch.cat([k_all[:, :, :sink, :], k_all[:, :, -self.sliding_window:, :]], dim=2)
            v_all = torch.cat([v_all[:, :, :sink, :], v_all[:, :, -self.sliding_window:, :]], dim=2)

        # Attention
        scale = 1.0 / math.sqrt(self.d_head)
        # If training (T>1 and no cache), causal SDPA is fine; with cache (T small) set is_causal=False
        is_causal = kv_cache is None
        y = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=None,
                                           dropout_p=self.dropout.p if self.training else 0.0,
                                           is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)

        # New cache should contain rotated k/v for all tokens so far
        if kv_cache is not None:
            k_new = torch.cat([kv_cache.k, k], dim=2)
            v_new = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_new, v_new = k, v
        new_cache = KVCache(k_new, v_new)
        return y, new_cache