from __future__ import annotations
from pathlib import Path
import torch

class ByteDataset:
    """Holds raw bytes of a text file and yields (x,y) blocks for LM.
    - block_size: sequence length (context window)
    - split: fraction for training (rest is val)
    """
    def __init__(self, path: str, block_size: int = 256, split: float = 0.9):
        data = Path(path).read_bytes()
        data = torch.tensor(list(data), dtype=torch.long)
        n = int(len(data) * split)
        self.train = data[:n]
        self.val = data[n:]
        self.block_size = block_size

    def get_batch(self, which: str, batch_size: int, device: torch.device):
        buf = self.train if which == 'train' else self.val
        assert len(buf) > self.block_size + 1, 'file too small for given block_size'
        ix = torch.randint(0, len(buf) - self.block_size - 1, (batch_size,))
        x = torch.stack([buf[i:i+self.block_size] for i in ix])
        y = torch.stack([buf[i+1:i+1+self.block_size] for i in ix])
        return x.to(device), y.to(device)