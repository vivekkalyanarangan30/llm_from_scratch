"""Visualize multi-head attention weights per head (grid)."""
import torch
from multi_head import MultiHeadSelfAttention
from vis_utils import save_attention_heads_grid

B, T, d_model, n_head = 1, 5, 12, 3
x = torch.randn(B, T, d_model)
attn = MultiHeadSelfAttention(d_model, n_head, trace_shapes=False)

out, w = attn(x)  # w: (B, H, T, T)

save_attention_heads_grid(w.detach().cpu().numpy(), filename="multi_head_attn_grid.png")