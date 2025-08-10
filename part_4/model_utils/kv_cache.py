from __future__ import annotations
import torch
from dataclasses import dataclass

@dataclass
class KVCache:
    k: torch.Tensor  # (B,H,T,D)
    v: torch.Tensor  # (B,H,T,D)

    @property
    def T(self):
        return self.k.size(2)

class RollingKV:
    """Rolling buffer with optional attention sink.
    Keeps first `sink` tokens + last `window` tokens.
    """
    def __init__(self, window: int, sink: int = 0):
        self.window = window
        self.sink = sink
        self.k = None
        self.v = None
    def step(self, k_new: torch.Tensor, v_new: torch.Tensor):
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        # crop
        if self.k.size(2) > self.window + self.sink:
            sink_part = self.k[:, :, :self.sink, :]
            sink_val  = self.v[:, :, :self.sink, :]
            tail_k = self.k[:, :, -self.window:, :]
            tail_v = self.v[:, :, -self.window:, :]
            self.k = torch.cat([sink_part, tail_k], dim=2)
            self.v = torch.cat([sink_val, tail_v], dim=2)
        return self.k, self.v