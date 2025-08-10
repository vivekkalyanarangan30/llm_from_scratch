from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

DEF_NAME = "model_last.pt"

# ----- helpers -----

def _extract_config_from_model(model) -> dict:
    """
    Best-effort extraction of GPTModern-like config.
    Falls back to {} for generic models (e.g., tests' Dummy()).
    """
    cfg = {}
    try:
        # Only proceed if model looks like your GPT
        tok_emb = getattr(model, "tok_emb", None)
        blocks = getattr(model, "blocks", None)
        if tok_emb is None or blocks is None or len(blocks) == 0:
            return cfg  # generic model → no config

        import torch.nn as nn
        try:
            from model_utils.swiglu import SwiGLU  # optional
        except Exception:
            class SwiGLU:  # dummy sentinel
                pass

        cfg["vocab_size"] = int(tok_emb.num_embeddings)
        cfg["block_size"]  = int(getattr(model, "block_size", 0) or 0)
        cfg["n_layer"]     = int(len(blocks))

        first_blk = blocks[0]
        attn = getattr(first_blk, "attn", None)
        if attn is None or not hasattr(attn, "n_head") or not hasattr(attn, "d_head"):
            return cfg  # partial info is fine

        cfg["n_head"] = int(attn.n_head)
        cfg["n_embd"] = int(attn.n_head * attn.d_head)
        # dropout if present
        drop = getattr(attn, "dropout", None)
        cfg["dropout"] = float(getattr(drop, "p", 0.0))

        cfg["use_rmsnorm"] = isinstance(getattr(model, "ln_f", None), nn.Identity)
        cfg["use_swiglu"]  = isinstance(getattr(first_blk, "ffn", None), SwiGLU)

        # optional extras
        for k in ("rope", "max_pos", "sliding_window", "attention_sink"):
            if hasattr(attn, k):
                cfg[k] = getattr(attn, k)
    except Exception:
        return {}  # never break save on extraction problems
    return cfg

def _verify_model_matches(model, cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """Return (ok, message)."""
    expected = {
        "block_size": cfg.get("block_size"),
        "n_layer":    cfg.get("n_layer"),
        "n_head":     cfg.get("n_head"),
        "n_embd":     cfg.get("n_embd"),
        "vocab_size": cfg.get("vocab_size"),
    }
    got = {
        "block_size": int(getattr(model, "block_size", -1)),
        "n_layer":    int(len(model.blocks)),
    }
    first_blk = model.blocks[0]
    got.update({
        "n_head":     int(first_blk.attn.n_head),
        "n_embd":     int(first_blk.attn.n_head * first_blk.attn.d_head),
        "vocab_size": int(model.tok_emb.num_embeddings),
    })
    diffs = [f"{k}: ckpt={expected[k]} vs model={got[k]}" for k in expected if expected[k] != got[k]]
    if diffs:
        return False, "Architecture mismatch:\n  " + "\n  ".join(diffs)
    return True, "ok"

def save_checkpoint(model, optimizer, scheduler, amp, step: int, out_dir: str,
                    tokenizer_dir: str | None = None, config: dict | None = None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = config if config is not None else _extract_config_from_model(model)

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
        "amp_scaler": amp.scaler.state_dict() if amp and getattr(amp, "scaler", None) else None,
        "step": int(step),
        **({"config": cfg} if cfg else {}),  # only write if we have something
    }, out / DEF_NAME)

    if tokenizer_dir is not None:
        (out / "tokenizer_dir.txt").write_text(tokenizer_dir)


def load_checkpoint(model, path: str, optimizer=None, scheduler=None, amp=None, strict: bool = True):
    ckpt = torch.load(path, map_location="cpu")

    cfg = ckpt.get("config")
    if cfg:  # only verify when we actually saved a config
        ok, msg = _verify_model_matches(model, cfg)  # same helper as before
        if not ok:
            raise RuntimeError(msg + "\nRebuild the model with the saved config, or load with strict=False.")

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(f"State dict mismatch:\n  missing: {missing}\n  unexpected: {unexpected}")

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler"])
    elif scheduler is not None and ckpt.get("scheduler") and hasattr(scheduler, "__dict__"):
        # best-effort legacy path (your original behavior)
        for k, v in ckpt["scheduler"].items():
            if k != "optimizer":
                setattr(scheduler, k, v)

    if amp is not None and ckpt.get("amp_scaler") is not None and getattr(amp, "scaler", None):
        amp.scaler.load_state_dict(ckpt["amp_scaler"])

    return ckpt.get("step", 0)