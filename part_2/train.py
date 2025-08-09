from __future__ import annotations
import argparse, time
import torch
from tokenizer import ByteTokenizer
from dataset import ByteDataset
from model_gpt import GPT


def estimate_loss(model: GPT, ds: ByteDataset, args) -> dict:
    model.eval()
    out = {}
    with torch.no_grad():
        for split in ['train', 'val']:
            losses = []
            for _ in range(args.eval_iters):
                xb, yb = ds.get_batch(split, args.batch_size, args.device)
                _, loss = model(xb, yb)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--out_dir', type=str, default='runs/min-gpt')
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--n_layer', type=int, default=4)
    p.add_argument('--n_head', type=int, default=4)
    p.add_argument('--n_embd', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--eval_interval', type=int, default=200)
    p.add_argument('--eval_iters', type=int, default=50)
    p.add_argument('--sample_every', type=int, default=200)
    p.add_argument('--sample_tokens', type=int, default=256)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top_k', type=int, default=50)
    p.add_argument('--top_p', type=float, default=None)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--amp', action='store_true')
    args = p.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    tok = ByteTokenizer()
    ds = ByteDataset(args.data, block_size=args.block_size)
    model = GPT(tok.vocab_size, args.block_size, args.n_layer, args.n_head, args.n_embd, args.dropout).to(args.device)

    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and args.device.type == 'cuda'))

    best_val = float('inf')
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        xb, yb = ds.get_batch('train', args.batch_size, args.device)
        with torch.cuda.amp.autocast(enabled=(args.amp and args.device.type == 'cuda')):
            _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()

        if step % 50 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | {(time.time()-t0):.1f}s")
            t0 = time.time()

        if step % args.eval_interval == 0:
            losses = estimate_loss(model, ds, args)
            print(f"eval | train {losses['train']:.4f} | val {losses['val']:.4f}")
            if losses['val'] < best_val:
                best_val = losses['val']
                ckpt_path = f"{args.out_dir}/model_best.pt"
                import os; os.makedirs(args.out_dir, exist_ok=True)
                torch.save({'model': model.state_dict(), 'config': {
                    'vocab_size': tok.vocab_size,
                    'block_size': args.block_size,
                    'n_layer': args.n_layer,
                    'n_head': args.n_head,
                    'n_embd': args.n_embd,
                    'dropout': args.dropout,
                }}, ckpt_path)
                print(f"saved checkpoint: {ckpt_path}")

        if args.sample_every > 0 and step % args.sample_every == 0:
            start = torch.randint(low=0, high=len(ds.train) - args.block_size - 1, size=(1,)).item()
            seed = ds.train[start:start + args.block_size].unsqueeze(0).to(args.device)
            out = model.generate(seed, max_new_tokens=args.sample_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
            txt = tok.decode(out[0].cpu())
            print("\n================ SAMPLE ================\n" + txt[-(args.block_size + args.sample_tokens):] + "\n=======================================\n")

    # final save
    import os; os.makedirs(args.out_dir, exist_ok=True)
    torch.save({'model': model.state_dict()}, f"{args.out_dir}/model_final.pt")


if __name__ == '__main__':
    main()