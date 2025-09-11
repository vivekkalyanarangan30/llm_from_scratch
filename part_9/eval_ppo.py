from __future__ import annotations
import argparse, torch
from pathlib import Path

from policy import PolicyWithValue
from rollout import RLHFTokenizer, sample_prompts, format_prompt_only

# Reward model
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_7'))
from model_reward import RewardModel  # noqa: E402


def score_policy(policy_ckpt: str, rm_ckpt: str, bpe_dir: str | None, n: int = 16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = RLHFTokenizer(block_size=256, bpe_dir=bpe_dir)

    ckpt = torch.load(policy_ckpt, map_location=device)
    cfg = ckpt.get('config', {})
    pol = PolicyWithValue(cfg.get('vocab_size', tok.vocab_size), cfg.get('block_size', tok.block_size),
                          cfg.get('n_layer', 2), cfg.get('n_head', 2), cfg.get('n_embd', 128)).to(device)
    pol.load_state_dict(ckpt['model'])
    pol.eval()

    # For comparing against reference policy (SFT)
    ref = PolicyWithValue(cfg.get('vocab_size', tok.vocab_size), cfg.get('block_size', tok.block_size),
                          cfg.get('n_layer', 2), cfg.get('n_head', 2), cfg.get('n_embd', 128)).to(device)
    ckpt_ref = torch.load("../part_6/runs/sft-demo/model_last.pt", map_location=device) # hardcoded path to SFT checkpoint
    ref.lm.load_state_dict(ckpt_ref['model']) 
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ref.eval()

    rckpt = torch.load(rm_ckpt, map_location=device)
    rm = RewardModel(vocab_size=rckpt['config'].get('vocab_size', tok.vocab_size), block_size=rckpt['config'].get('block_size', tok.block_size),
                     n_layer=rckpt['config'].get('n_layer', 4), n_head=rckpt['config'].get('n_head', 4), n_embd=rckpt['config'].get('n_embd', 256)).to(device)
    rm.load_state_dict(rckpt['model'])
    rm.eval()

    prompts = sample_prompts(n)
    rewards = []
    for p in prompts:
        prefix = format_prompt_only(p).replace('</s>', '')
        ids = tok.encode(prefix)
        x = torch.tensor([ids[-tok.block_size:]], dtype=torch.long, device=device)
        with torch.no_grad():
            y = pol.generate(x, max_new_tokens=128, temperature=0.2, top_k=50)
            y_old = ref.generate(x, max_new_tokens=128, temperature=0.2, top_k=50)
        resp = tok.decode(y[0].tolist()[len(ids[-tok.block_size:]):])
        resp_old = tok.decode(y_old[0].tolist()[len(ids[-tok.block_size:]):])

        # compute RM reward on formatted full text
        from part_6.formatters import Example, format_example
        text = format_example(Example(p, resp))
        z = torch.tensor([tok.encode(text)[:tok.block_size]], dtype=torch.long, device=device)
        with torch.no_grad():
            r = rm(z)[0].item()
        rewards.append(r)
    return sum(rewards)/max(1,len(rewards))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--policy_ckpt', type=str, required=True)
    p.add_argument('--reward_ckpt', type=str, required=True)
    p.add_argument('--split', type=str, default='val[:32]')  # unused in this tiny script
    p.add_argument('--bpe_dir', type=str, default=None)
    args = p.parse_args()

    avg_r = score_policy(args.policy_ckpt, args.reward_ckpt, args.bpe_dir, n=16)
    print(f"Avg RM reward: {avg_r:.4f}")