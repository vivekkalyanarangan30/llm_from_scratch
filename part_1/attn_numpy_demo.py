"""1.2 Self-attention from first principles on a tiny example (NumPy only).
We use T=3 tokens, d_model=4, d_k=d_v=2, single-head.
This script prints intermediate tensors so you can trace the math.

Dimensions summary (single head)
--------------------------------
X:          (B=1, T=3, d_model=4)
Wq/Wk/Wv:   (d_model=4, d_k=2)
Q,K,V:      (1, 3, 2)
Scores:     (1, 3, 3)   = Q @ K^T
Weights:    (1, 3, 3)   = softmax over last dim
Output:     (1, 3, 2)   = Weights @ V
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# Toy inputs (batch=1, seq=3, d_model=4)
X = np.array([[[0.1, 0.2, 0.3, 0.4],
               [0.5, 0.4, 0.3, 0.2],
               [0.0, 0.1, 0.0, 0.1]]], dtype=np.float32)

# Weight matrices (learned in real models). We fix numbers for determinism.
Wq = np.array([[ 0.2, -0.1],
               [ 0.0,  0.1],
               [ 0.1,  0.2],
               [-0.1,  0.0]], dtype=np.float32)
Wk = np.array([[ 0.1,  0.1],
               [ 0.0, -0.1],
               [ 0.2,  0.0],
               [ 0.0,  0.2]], dtype=np.float32)
Wv = np.array([[ 0.1,  0.0],
               [-0.1,  0.1],
               [ 0.2, -0.1],
               [ 0.0,  0.2]], dtype=np.float32)

# Project to Q, K, V
Q = X @ Wq  # (1,3,2)
K = X @ Wk  # (1,3,2)
V = X @ Wv  # (1,3,2)

print("Q shape:", Q.shape, "\nQ=\n", Q[0])
print("K shape:", K.shape, "\nK=\n", K[0])
print("V shape:", V.shape, "\nV=\n", V[0])

# Scaled dot-products
scale = 1.0 / np.sqrt(Q.shape[-1])
attn_scores = (Q @ K.transpose(0,2,1)) * scale  # (1,3,3)

# Causal mask (upper triangle set to -inf so softmax->0)
mask = np.triu(np.ones((1,3,3), dtype=bool), k=1)
attn_scores = np.where(mask, -1e9, attn_scores)

# Softmax over last dim
weights = np.exp(attn_scores - attn_scores.max(axis=-1, keepdims=True))
weights = weights / weights.sum(axis=-1, keepdims=True)
print("Weights shape:", weights.shape, "\nAttention Weights (causal)=\n", weights[0])

# Weighted sum of V
out = weights @ V  # (1,3,2)
print("Output shape:", out.shape, "\nOutput=\n", out[0])