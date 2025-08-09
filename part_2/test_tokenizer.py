import torch
from tokenizer import ByteTokenizer

def test_roundtrip():
    tok = ByteTokenizer()
    s = "Hello, ByteTok! äö"
    ids = tok.encode(s)
    assert ids.dtype == torch.long
    s2 = tok.decode(ids)
    assert len(s2) > 0