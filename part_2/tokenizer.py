from __future__ import annotations
import torch

class ByteTokenizer:
    """Ultra-simple byte-level tokenizer.
    - encode(str) -> LongTensor [N]
    - decode(Tensor[int]) -> str
    - vocab_size = 256
    """
    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor(list(s.encode('utf-8')), dtype=torch.long)

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return bytes(ids).decode('utf-8', errors='ignore')

    @property
    def vocab_size(self) -> int:
        return 256