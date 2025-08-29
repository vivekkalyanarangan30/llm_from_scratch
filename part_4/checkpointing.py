from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]/'part_3'))

import torch
import shutil

DEF_NAME = "model_last.pt"

# ----------------------------- TB-only helpers (safe no-ops otherwise) ----------------------------- #
def _is_tb(logger) -> bool:
    return hasattr(logger, "hist")

def _log_hparams_tb(logger, args, total_steps):
    if not _is_tb(logger): return
    try:
        h = dict(
            vocab_size=args.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
            n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout, lr=args.lr,
            warmup_steps=args.warmup_steps, batch_size=args.batch_size, grad_accum=args.grad_accum_steps,
            mixed_precision=args.mixed_precision, steps=args.steps, epochs=args.epochs,
        )
        logger.hparams(h, {"meta/total_steps": float(total_steps)})
    except Exception:
        pass

def _maybe_log_graph_tb(logger, model, xb, yb):
    if not hasattr(logger, "graph"): 
        return
    try:
        class _TensorOnly(nn.Module):
            def __init__(self, m): 
                super().__init__(); self.m = m.eval()
            def forward(self, x, y=None):
                out = self.m(x, y) if y is not None else self.m(x)
                if isinstance(out, (list, tuple)):
                    for o in out:
                        if torch.is_tensor(o):
                            return o
                    return out[0]
                return out
        wrapped = _TensorOnly(model).to(xb.device)
        logger.graph(wrapped, (xb, yb))
    except Exception:
        pass

def _log_model_stats(logger, model, step: int, do_hists: bool = False):
    if not _is_tb(logger): return
    try:
        params = [p for p in model.parameters() if p.requires_grad]
        total_param_norm = torch.norm(torch.stack([p.detach().norm(2) for p in params]), 2).item()
        grads = [p.grad for p in params if p.grad is not None]
        total_grad_norm = float('nan')
        if grads:
            total_grad_norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]), 2).item()
        logger.log(step=step, **{
            "train/param_global_l2": total_param_norm,
            "train/grad_global_l2": total_grad_norm,
        })
        if do_hists:
            for name, p in model.named_parameters():
                logger.hist(f"params/{name}", p, step)
                if p.grad is not None:
                    logger.hist(f"grads/{name}", p.grad, step)
    except Exception:
        pass

def _maybe_log_attention(logger, model, step: int, every: int = 100, window: int = 2):
    if not _is_tb(logger): return
    if step == 0 or (step % every): return
    try:
        with torch.no_grad():
            for name, m in model.named_modules():
                p = getattr(m, "last_attn", None)  # [B,H,Tq,Tk]
                if p is None: continue
                eps = 1e-12
                p_eps = p.clamp_min(eps)
                ent = (-p_eps * p_eps.log()).sum(dim=-1).mean().item()
                B,H,Tq,Tk = p.shape
                i = torch.arange(Tq, device=p.device).unsqueeze(-1)
                j = torch.arange(Tk, device=p.device).unsqueeze(0)
                diag = ((p * ((j - i).abs() <= window)).sum(dim=-1)).mean().item()
                logger.log(step=step, **{
                    f"attn/{name}/entropy": ent,
                    f"attn/{name}/diagonal_mass_w{window}": diag
                })
    except Exception:
        pass

def _log_runtime(logger, step: int, it_t0: float, xb, device):
    try:
        dt = time.time() - it_t0
        toks = int(xb.numel())
        toks_per_s = toks / max(dt, 1e-6)
        mem = torch.cuda.memory_allocated()/(1024**2) if torch.cuda.is_available() else 0.0
        logger.log(step=step, **{
            "sys/throughput_tokens_per_s": toks_per_s,
            "sys/step_time_s": dt,
            "sys/gpu_mem_alloc_mb": mem
        })
    except Exception:
        pass

def _log_samples_tb(logger, model, tok, xb, device, step: int, max_new_tokens: int = 64):
    if not _is_tb(logger): return
    if tok is None: return
    try:
        model.eval()
        with torch.no_grad():
            out = model.generate(xb[:1].to(device), max_new_tokens=max_new_tokens, temperature=1.0, top_k=50)
        model.train()
        text = tok.decode(out[0].tolist())
        logger.text("samples/generation", text, step)
    except Exception:
        pass
# ---------------------------------------------------------------------- #

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
            from swiglu import SwiGLU  # optional
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

# ----------------------------- checkpoint/save utils ----------------------------- #
def checkpoint_paths(out_dir: Path, step: int):
    return out_dir / f"model_step{step:07d}.pt", out_dir / "model_last.pt"

def atomic_save_all(model, optim, sched, amp, step: int, out_dir: Path,
                    tok_dir: str | None, keep_last_k: int, config: dict):
    """Write model_last.pt (with config) + a rolling per-step copy."""
    save_checkpoint(model, optim, sched, amp, step, str(out_dir), tok_dir, config=config)  # writes model_last.pt
    per_step, last = checkpoint_paths(out_dir, step)
    try:
        shutil.copy2(last, per_step)
    except Exception:
        pass
    # GC old per-step checkpoints
    try:
        ckpts = sorted(out_dir.glob("model_step*.pt"))
        for old in ckpts[:-keep_last_k]:
            old.unlink(missing_ok=True)
    except Exception:
        pass