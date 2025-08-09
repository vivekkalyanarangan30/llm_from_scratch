from __future__ import annotations
import argparse, time, torch
from model_modern import GPTModern
from tokenizer import ByteTokenizer


def time_generate(model, prompt_len=64, tokens=256, device='cpu'):
    tok = ByteTokenizer()
    prompt = torch.randint(0, tok.vocab_size, (1, prompt_len), device=device)
    t0 = time.time()
    with torch.no_grad():
        _ = model.generate(prompt, max_new_tokens=tokens)
    return time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu')
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--n_layer', type=int, default=4)
    p.add_argument('--n_head', type=int, default=4)
    p.add_argument('--n_embd', type=int, default=256)
    p.add_argument('--rmsnorm', action='store_true')
    p.add_argument('--swiglu', action='store_true')
    p.add_argument('--rope', action='store_true')
    args = p.parse_args()

    device = torch.device(args.device)

    model = GPTModern(vocab_size=256, block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head,
                      n_embd=args.n_embd, use_rmsnorm=args.rmsnorm, use_swiglu=args.swiglu, rope=args.rope).to(device)
    secs = time_generate(model, device=device)
    print(f"gen time: {secs:.3f}s | rmsnorm={args.rmsnorm} swiglu={args.swiglu} rope={args.rope}")


if __name__ == '__main__':
    main()