import argparse, torch
from moe import MoE

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', type=int, default=64)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--experts', type=int, default=4)
    p.add_argument('--top_k', type=int, default=1)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    x = torch.randn(2, args.tokens//2, args.hidden, device=device)  # (B=2,T=tokens/2,C)

    moe = MoE(dim=args.hidden, n_expert=args.experts, k=args.top_k).to(device)
    with torch.no_grad():
        y, aux = moe(x)

    # simple routing histogram
    from gating import TopKGate
    gate = moe.gate
    idx, w, _ = gate(x.view(-1, args.hidden))
    hist = torch.bincount(idx[:,0], minlength=args.experts)
    print(f"Output shape: {tuple(y.shape)} | aux={float(aux):.4f}")
    print("Primary expert load (counts):", hist.tolist())