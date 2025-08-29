from __future__ import annotations
import argparse, time, os, signal, re, shutil
from pathlib import Path

import torch
import torch.nn as nn

import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))

from model_modern import GPTModern

from tokenizer_bpe import BPETokenizer
from dataset_bpe import make_loader
from lr_scheduler import WarmupCosineLR
from amp_accum import AmpGrad
from checkpointing import (
    load_checkpoint,
    _log_hparams_tb,
    _maybe_log_graph_tb,
    _is_tb,
    _log_model_stats,
    _maybe_log_attention,
    _log_samples_tb,
    _log_runtime,
    atomic_save_all
    )
from logger import init_logger


# ----------------------------- model helpers ----------------------------- #
def build_model(vocab_size: int, block_size: int, n_layer: int, n_head: int, n_embd: int, dropout: float):
    return GPTModern(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd, dropout=dropout,
        use_rmsnorm=True, use_swiglu=True, rope=True
    )

def infer_cfg_from_state(sd: dict, fallback_vocab: int | None = None) -> dict:
    """Best-effort inference for old checkpoints without a config."""
    V, C = sd["tok_emb.weight"].shape
    block_size = sd.get("pos_emb.weight", torch.empty(256, C)).shape[0]
    layer_ids = {int(m.group(1)) for k in sd.keys() if (m := re.match(r"blocks\.(\d+)\.", k))}
    n_layer = max(layer_ids) + 1 if layer_ids else 1
    has_swiglu = any(".ffn.w1.weight" in k for k in sd.keys())
    # pick a head count that divides C
    n_head = None
    for h in (16, 8, 4, 2, 1):
        if C % h == 0:
            n_head = h; break
    if n_head is None: n_head = 1
    return dict(
        vocab_size=fallback_vocab or V,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=C,
        dropout=0.0,
        use_rmsnorm=True,
        use_swiglu=has_swiglu,
        rope=True,
        max_pos=4096,
        sliding_window=None,
        attention_sink=0,
    )

def run_cfg_from_args(args, vocab_size: int) -> dict:
    return dict(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        use_rmsnorm=True,
        use_swiglu=True,
        rope=True,
        max_pos=4096,
        sliding_window=None,
        attention_sink=0,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--out', type=str, default='runs/part4')
    p.add_argument('--bpe', action='store_true')
    p.add_argument('--vocab_size', type=int, default=32000)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--steps', type=int, default=300, help='max steps (cap per run)')
    p.add_argument('--n_layer', type=int, default=6)
    p.add_argument('--n_head', type=int, default=8)
    p.add_argument('--n_embd', type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup_steps', type=int, default=20)
    p.add_argument('--mixed_precision', action='store_true')
    p.add_argument('--grad_accum_steps', type=int, default=4)
    p.add_argument('--log', choices=['wandb','tensorboard','none'], default='none')
    p.add_argument('--save_every', type=int, default=50, help='save checkpoint every N optimizer steps')
    p.add_argument('--keep_last_k', type=int, default=2, help='keep last K step checkpoints (plus model_last.pt)')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "model_last.pt"
    have_ckpt = ckpt_path.exists()

    # Pre-read ckpt meta if present (config + tokenizer dir)
    ckpt = None
    ckpt_cfg = {}
    saved_tok_dir = None
    if have_ckpt:
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        ckpt_cfg = ckpt.get("config") or {}
        tok_file = ckpt_path.with_name("tokenizer_dir.txt")
        saved_tok_dir = tok_file.read_text().strip() if tok_file.exists() else None

    # --- tokenizer ---
    tok = None
    tok_dir = None
    if saved_tok_dir is not None:
        tok = BPETokenizer(); tok.load(saved_tok_dir)
        tok_dir = saved_tok_dir
        vocab_size = tok.vocab_size
        print(f"[resume] Loaded tokenizer from {tok_dir} (vocab={vocab_size})")
    elif args.bpe:
        tok = BPETokenizer(vocab_size=args.vocab_size)
        tok.train(args.data)
        tok_dir = str(out_dir / 'tokenizer')
        Path(tok_dir).mkdir(parents=True, exist_ok=True)
        tok.save(tok_dir)
        vocab_size = tok.vocab_size
        print(f"[init] Trained tokenizer to {tok_dir} (vocab={vocab_size})")
    else:
        vocab_size = 256  # byte fallback

    # --- data ---
    train_loader = make_loader(args.data, tok, args.block_size, args.batch_size, shuffle=True)

    # --- model/opt/sched ---
    # Build config for this run (from ckpt if present, else from args). If ckpt has no config, infer.
    if have_ckpt:
        sd = ckpt["model"]
        if ckpt_cfg:
            # if tokenizer present and vocab differs, override
            if tok is not None and ckpt_cfg.get("vocab_size") != vocab_size:
                ckpt_cfg = {**ckpt_cfg, "vocab_size": vocab_size}
            cfg_build = {**infer_cfg_from_state(sd, vocab_size), **ckpt_cfg}  # ckpt_cfg wins
        else:
            cfg_build = infer_cfg_from_state(sd, vocab_size)
    else:
        cfg_build = run_cfg_from_args(args, vocab_size)

    model = GPTModern(**cfg_build).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    total_steps = min(args.steps, args.epochs * len(train_loader))
    sched = WarmupCosineLR(optim, warmup_steps=min(args.warmup_steps, total_steps//10 or 1),
                           total_steps=total_steps, base_lr=args.lr)
    amp = AmpGrad(optim, accum=args.grad_accum_steps, amp=args.mixed_precision)

    # --- resume if possible ---
    step = 0
    if have_ckpt:
        try:
            step = load_checkpoint(model, str(ckpt_path), optimizer=optim, scheduler=sched, amp=amp, strict=True)
            print(f"[resume] Loaded checkpoint at step {step}")
        except Exception as e:
            print(f"[resume] Strict load failed ({e}); trying non-strict weights only.")
            model.load_state_dict(ckpt["model"], strict=False)

    # --- logger ---
    logger = init_logger(args.log, out_dir=str(out_dir))
    _log_hparams_tb(logger, args, total_steps)

    # optional: log graph once if TB
    if _is_tb(logger):
        try:
            ex_x, ex_y = next(iter(train_loader))
            _maybe_log_graph_tb(logger, model, ex_x.to(device), ex_y.to(device))
        except Exception:
            pass

    # --- signal handler (graceful save on Ctrl-C/SIGTERM) ---
    save_requested = {"flag": False}
    def _on_term(sig, frame):
        save_requested["flag"] = True
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT,  _on_term)

    # --- train loop ---
    model.train()
    while step < args.steps:
        for xb, yb in train_loader:
            if step >= args.steps: break
            if save_requested["flag"]:
                atomic_save_all(model, optim, sched, amp, step, out_dir, tok_dir, args.keep_last_k, cfg_build)
                print(f"[signal] Saved checkpoint at step {step} to {out_dir}. Exiting.")
                return

            it_t0 = time.time()
            xb, yb = xb.to(device), yb.to(device)
            with torch.cuda.amp.autocast(enabled=amp.amp):
                logits, loss, _ = model(xb, yb)
            amp.backward(loss)
            if amp.should_step():
                amp.step(); amp.zero_grad()
                lr = sched.step()
                step += 1

                # periodic checkpoint
                if step % args.save_every == 0:
                    atomic_save_all(model, optim, sched, amp, step, out_dir, tok_dir, args.keep_last_k, cfg_build)
                    if _is_tb(logger): logger.text("meta/checkpoint", f"Saved at step {step}", step)

                # base metrics (works for all loggers)
                if step % 50 == 0:
                    logger.log(step=step, loss=float(loss.item()), lr=float(lr))
                    _log_runtime(logger, step, it_t0, xb, device)

                    # TB-only richer stuff
                    _log_model_stats(logger, model, step, do_hists=False)
                    _maybe_log_attention(logger, model, step, every=100, window=2)
                    _log_samples_tb(logger, model, tok, xb, device, step, max_new_tokens=64)

    # final save
    atomic_save_all(model, optim, sched, amp, step, out_dir, tok_dir, args.keep_last_k, cfg_build)
    print(f"Saved checkpoint to {out_dir}/model_last.pt")


if __name__ == '__main__':
    main()
