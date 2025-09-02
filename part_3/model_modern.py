from __future__ import annotations
import torch
import torch.nn as nn
from block_modern import TransformerBlockModern
from tokenizer import ByteTokenizer

# Get the absolute path to the folder that contains part_2 and part_3
import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

class GPTModern(nn.Module):
    def __init__(self, vocab_size: int = 256, block_size: int = 256,
                 n_layer: int=4, n_head: int=4, n_embd: int=256, dropout: float=0.0,
                 use_rmsnorm: bool = True, use_swiglu: bool = True, rope: bool = True,
                 max_pos: int = 4096, sliding_window: int | None = None, attention_sink: int = 0, n_kv_head: int | None = None):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        # self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlockModern(n_embd, n_head, dropout, use_rmsnorm, use_swiglu, rope, max_pos, sliding_window, attention_sink, n_kv_head)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.Identity() if use_rmsnorm else nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, kv_cache_list=None, start_pos: int = 0):
        B, T = idx.shape
        assert T <= self.block_size
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) 
        # + self.pos_emb(pos)
        x = self.drop(x)

        new_caches = []
        for i, blk in enumerate(self.blocks):
            cache = None if kv_cache_list is None else kv_cache_list[i]
            x, cache = blk(x, kv_cache=cache, start_pos=start_pos)
            new_caches.append(cache)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            import torch.nn.functional as F
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, new_caches

    @torch.no_grad()
    def generate(self, 
                 prompt: torch.Tensor, 
                 max_new_tokens=200, 
                 temperature=1.0, 
                 top_k=50, 
                 top_p=None,
                 eos_id=1, # addition from part 6 for early stopping
                 sliding_window: int | None = None, 
                 attention_sink: int = 0):
        try:
            from utils import top_k_top_p_filtering as _tk
        except Exception:
            _tk = lambda x, **_: x

        self.eval()
        idx = prompt
        kvs = [None] * len(self.blocks)

        for _ in range(max_new_tokens):
            # feed full prompt once; then only the last token
            idx_cond = idx[:, -self.block_size:] if kvs[0] is None else idx[:, -1:]

            # absolute start position from cache length (0 on first step)
            start_pos = 0 if kvs[0] is None else kvs[0].k.size(2)

            logits, _, kvs = self(idx_cond, kv_cache_list=kvs, start_pos=start_pos)

            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            next_logits = _tk(next_logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.argmax(probs, dim=-1, keepdim=True) if temperature == 0.0 else torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)

            # addition from part 6 for early stopping
            if eos_id is not None:
                if (next_id == eos_id).all():
                    break

        return idx


    @torch.no_grad()
    def generate_nocache(self, prompt: torch.Tensor, max_new_tokens=200, temperature=1.0, top_k=50, top_p=None,
                sliding_window: int | None = None, attention_sink: int = 0):
        try:
            from utils import top_k_top_p_filtering as _tk
        except Exception:
            _tk = lambda x, **_: x

        self.eval()
        idx = prompt

        for _ in range(max_new_tokens):
            # always run a full forward over the cropped window, with NO cache
            idx_cond = idx[:, -self.block_size:]
            # absolute position of first token in the window (matches cached path)
            start_pos = idx.size(1) - idx_cond.size(1)

            logits, _, _ = self(idx_cond, kv_cache_list=None, start_pos=start_pos)

            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            next_logits = _tk(next_logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(next_logits, dim=-1)
            topv, topi = torch.topk(probs, 10)
            print("top ids:", topi.tolist())
            print("top vs:", topv.tolist())
            next_id = torch.argmax(probs, dim=-1, keepdim=True) if temperature == 0.0 else torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx

