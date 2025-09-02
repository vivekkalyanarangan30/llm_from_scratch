from __future__ import annotations
import argparse, torch
import torch.nn as nn
from pathlib import Path
torch.manual_seed(0)

# Reuse GPTModern from Part 3
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))
from model_modern import GPTModern  # noqa: E402

from dataset_sft import load_tiny_hf
from collator_sft import SFTCollator
from curriculum import LengthCurriculum


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='huggingface', help='huggingface or path to local jsonl (unused in demo)')
    p.add_argument('--ckpt', type=str, required=False)
    p.add_argument('--out', type=str, default='runs/sft')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--n_layer', type=int, default=4)
    p.add_argument('--n_head', type=int, default=4)
    p.add_argument('--n_embd', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--bpe_dir', type=str, default='../part_4/runs/part4-demo/tokenizer') # assumes tokenizer exists from Part 4
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Load a tiny HF slice or fallback examples
    items = load_tiny_hf(split='train[:24]', sample_dataset=False)

    # Print few samples
    print(f"Loaded {len(items)} SFT items. Few samples:")
    for it in items[:3]:
        print(f"PROMPT: {it.prompt}\nRESPONSE: {it.response}\n{'-'*40}")

    # Curriculum over (prompt,response)
    tuples = [(it.prompt, it.response) for it in items]
    cur = list(LengthCurriculum(tuples))
    print(cur)

    # Collator + model
    col = SFTCollator(block_size=args.block_size, bpe_dir=args.bpe_dir)
    model = GPTModern(vocab_size=col.vocab_size, block_size=args.block_size,
                      n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                      use_rmsnorm=True, use_swiglu=True, rope=True).to(device)

    if args.ckpt:
        print(f"Using model config from checkpoint {args.ckpt}")
        ckpt = torch.load(args.ckpt, map_location=device)
        cfg = ckpt.get('config', {})
        model.load_state_dict(ckpt['model'])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    model.train()

    # Simple loop (single machine). We just cycle curriculum to fill batches, for a few steps.
    step = 0
    i = 0
    while step < args.steps:
        batch = cur[i:i+args.batch_size]
        if not batch:
            # restart curriculum
            # cur = list(LengthCurriculum(tuples)); 
            i = 0
            continue
        xb, yb = col.collate(batch)
        xb, yb = xb.to(device), yb.to(device)
        logits, loss, _ = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1; i += args.batch_size
        if step % 20 == 0:
            print(f"step {step}: loss={loss.item():.4f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    cfg = {
        "vocab_size": col.vocab_size,
        "block_size": args.block_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "dropout": 0.0,
        "use_rmsnorm": True,
        "use_swiglu": True,
        "rope": True,
        # tokenizer info (best-effort)
        "tokenizer_type": "byte" if col.vocab_size == 256 else "bpe",
        "tokenizer_dir": None,   # set a real path if you have a trained BPE dir
    }
    torch.save({'model': model.state_dict(), 'config': cfg},
               str(Path(args.out)/'model_last.pt'))
    print(f"Saved SFT checkpoint to {args.out}/model_last.pt")

if __name__ == '__main__':
    main()