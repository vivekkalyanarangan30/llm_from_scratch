import torch
from dataset import ByteDataset


def test_shift_alignment(tmp_path):
    p = tmp_path / 'toy.txt'
    p.write_text('abcdefg')
    ds = ByteDataset(str(p), block_size=3, split=1.0)
    x, y = ds.get_batch('train', 2, device=torch.device('cpu'))
    # shift must be next-token
    assert (y[:, :-1] == x[:, 1:]).all()
