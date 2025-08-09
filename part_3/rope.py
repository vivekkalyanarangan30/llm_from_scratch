from __future__ import annotations
import torch
import math

class RoPECache:
    """Precompute cos/sin for positions up to max_pos for even head_dim."""
    def __init__(self, head_dim: int, max_pos: int, base: float = 10000.0, device: torch.device | None = None):
        assert head_dim % 2 == 0, "RoPE head_dim must be even"
        self.head_dim = head_dim
        self.max_pos = max_pos
        self.device = device
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        t = torch.arange(max_pos, device=device).float()
        freqs = torch.outer(t, inv_freq)  # (max_pos, head_dim/2)
        self.cos = torch.cos(freqs)       # (max_pos, D/2)
        self.sin = torch.sin(freqs)       # (max_pos, D/2)
    def get(self, positions: torch.Tensor):
        # positions: (T,) or (1,T)
        if positions.dim() == 2:
            positions = positions[0]
        need = int(positions.max().item()) + 1 if positions.numel() > 0 else 1
        if need > self.max_pos:
            # grow tables
            self._build(max(need, int(self.max_pos * 2)))
        cos = self.cos[positions]  # (T, D/2)
        sin = self.sin[positions]
        return cos, sin

def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Rotate pairs of last-dim features for q,k.
    q,k: (B,H,T,D) with D even
    cos,sin: (T,D/2) → broadcast to (1,1,T,D/2)
    """
    B,H,T,D = q.shape
    assert D % 2 == 0
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,D/2)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def _rotate(x):
        x1 = x[..., ::2]   # (B,H,T,D/2)
        x2 = x[..., 1::2]
        xr1 = x1 * cos - x2 * sin
        xr2 = x1 * sin + x2 * cos
        out = torch.empty_like(x)
        out[..., ::2] = xr1
        out[..., 1::2] = xr2
        return out

    return _rotate(q), _rotate(k)