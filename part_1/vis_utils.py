import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), 'out')


def _ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)


def save_matrix_heatmap(mat: np.ndarray, title: str, filename: str, xlabel: str = '', ylabel: str = ''):
    """Generic matrix heatmap saver.
    Do not set any specific colors/styles; keep defaults for clarity.
    """
    _ensure_out()
    plt.figure()
    plt.imshow(mat, aspect='auto')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def save_attention_heads_grid(weights: np.ndarray, filename: str, title_prefix: str = "Head"):
    """Plot all heads in a single grid figure (B=1 assumed).
    weights: (1, H, T, T)
    """
    _ensure_out()
    _, H, T, _ = weights.shape
    cols = min(4, H)
    rows = (H + cols - 1) // cols
    plt.figure(figsize=(3*cols, 3*rows))
    for h in range(H):
        ax = plt.subplot(rows, cols, h+1)
        ax.imshow(weights[0, h], aspect='auto')
        ax.set_title(f"{title_prefix} {h}")
        ax.set_xlabel('Key pos')
        ax.set_ylabel('Query pos')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")