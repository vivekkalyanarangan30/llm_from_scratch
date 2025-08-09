from __future__ import annotations
import argparse, torch
from dataset import ByteDataset
from model_gpt import GPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--iters', type=int, default=100)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    ds = ByteDataset(args.data, block_size=args.block_size)
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', {
        'vocab_size': 256,
        'block_size': args.block_size,
        'n_layer': 4,
        'n_head': 4,
        'n_embd': 256,
        'dropout': 0.0,
    })
    model = GPT(**cfg).to(device)
    model.load_state_dict(ckpt['model'])

    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(args.iters):
            xb, yb = ds.get_batch('val', args.batch_size, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
    print(f"val loss: {sum(losses)/len(losses):.4f}")


if __name__ == '__main__':
    main()