from __future__ import annotations
import argparse, torch

# Reuse GPTModern
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_3'))
from model_utils.model_modern import GPTModern  # noqa: E402

from collator_sft import SFTCollator
from formatters import format_prompt_only


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--prompt', type=str, required=True)
    p.add_argument('--tokens', type=int, default=80)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--bpe_dir', type=str, default='../part_4/runs/part4-demo/tokenizer')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get('config', {})

    col = SFTCollator(block_size=cfg.get('block_size', 256), bpe_dir=args.bpe_dir)
    model = GPTModern(vocab_size=col.vocab_size, block_size=cfg.get('block_size', 256)).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    prompt_text = format_prompt_only(args.prompt)
    ids = col.encode(prompt_text)
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=args.tokens, temperature=0.8, top_k=50)

    # decode: prefer BPE if collator has it, else fall back to bytes
    out_ids = out[0].tolist()
    orig_len = idx.size(1)
    if hasattr(col, "tok") and hasattr(col.tok, "decode"):
        # decode full text or just the generated suffix; suffix is often clearer
        generated = col.tok.decode(out_ids)
        print(generated)
    else:
        generated = bytes(out_ids[orig_len:]).decode("utf-8", errors="ignore")
        print(generated)


if __name__ == '__main__':
    main()