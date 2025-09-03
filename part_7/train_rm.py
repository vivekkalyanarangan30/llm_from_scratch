from __future__ import annotations
import argparse, torch
from pathlib import Path

from data_prefs import load_preferences
from collator_rm import PairCollator
from model_reward import RewardModel
from loss_reward import bradley_terry_loss, margin_ranking_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='runs/rm-demo')
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--n_layer', type=int, default=4)
    p.add_argument('--n_head', type=int, default=4)
    p.add_argument('--n_embd', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--loss', choices=['bt','margin'], default='bt')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--bpe_dir', type=str, default=None)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # data
    items = load_preferences(split='train[:80]')
    triples = [(it.prompt, it.chosen, it.rejected) for it in items]

    # collator + model
    col = PairCollator(block_size=args.block_size, bpe_dir=args.bpe_dir)
    model = RewardModel(vocab_size=col.vocab_size, block_size=args.block_size,
                        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # train (tiny)
    step = 0; i = 0
    while step < args.steps:
        batch = triples[i:i+args.batch_size]
        if not batch:
            i = 0; continue
        pos, neg = col.collate(batch)
        pos, neg = pos.to(device), neg.to(device)
        r_pos = model(pos)
        r_neg = model(neg)
        if args.loss == 'bt':
            loss = bradley_terry_loss(r_pos, r_neg)
        else:
            loss = margin_ranking_loss(r_pos, r_neg, margin=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1; i += args.batch_size
        if step % 25 == 0:
            acc = (r_pos > r_neg).float().mean().item()
            print(f"step {step}: loss={loss.item():.4f} acc={acc:.2f}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'config': {
        'vocab_size': col.vocab_size,
        'block_size': args.block_size,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'n_embd': args.n_embd,
    }}, str(Path(args.out)/'model_last.pt'))
    print(f"Saved reward model to {args.out}/model_last.pt")

if __name__ == '__main__':
    main()