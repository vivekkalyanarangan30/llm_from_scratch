from __future__ import annotations
import argparse, torch
from pathlib import Path

# import torch
# torch.manual_seed(0)

from policy import PolicyWithValue
from rollout import RLHFTokenizer, format_prompt_only, format_example, sample_prompts, gather_logprobs, shift_labels
from rollout import model_logprobs

# Reward model from Part 7
import sys
from pathlib import Path as _P
sys.path.append(str(_P(__file__).resolve().parents[1]/'part_7'))
from model_reward import RewardModel  # noqa: E402

from ppo_loss import ppo_losses


def compute_reward(reward_model: RewardModel, tok: RLHFTokenizer, prompt: str, response: str, device) -> float:
    text = format_example(__import__('part_6.formatters', fromlist=['Example']).Example(prompt, response))
    ids = tok.encode(text)
    x = torch.tensor([ids[:tok.block_size]], dtype=torch.long, device=device)
    with torch.no_grad():
        r = reward_model(x)
    return float(r[0].item())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='runs/ppo-demo')
    p.add_argument('--policy_ckpt', type=str, required=True, help='SFT checkpoint (Part 6)')
    p.add_argument('--reward_ckpt', type=str, required=True, help='Reward model checkpoint (Part 7)')
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--block_size', type=int, default=256)
    p.add_argument('--resp_len', type=int, default=64)
    p.add_argument('--kl_coef', type=float, default=0.01)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--lam', type=float, default=0.95)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--bpe_dir', type=str, default=None)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # tokenizer
    tok = RLHFTokenizer(block_size=args.block_size, bpe_dir=args.bpe_dir)

    # Load SFT policy as initial policy AND reference
    ckpt = torch.load(args.policy_ckpt, map_location=device)
    cfg = ckpt.get('config', {})
    vocab_size = cfg.get('vocab_size', tok.vocab_size)
    block_size = cfg.get('block_size', tok.block_size)
    n_layer = cfg.get('n_layer', 2)
    n_head  = cfg.get('n_head', 2)
    n_embd  = cfg.get('n_embd', 128)

    policy = PolicyWithValue(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    policy.lm.load_state_dict(ckpt['model'])  # initialize LM weights from SFT


    ref = PolicyWithValue(vocab_size, block_size, n_layer, n_head, n_embd).to(device)
    ref.lm.load_state_dict(ckpt['model'])
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    ref.eval()

    # Reward model
    rckpt = torch.load(args.reward_ckpt, map_location=device)
    rm = RewardModel(vocab_size=rckpt['config'].get('vocab_size', tok.vocab_size), block_size=rckpt['config'].get('block_size', tok.block_size),
                     n_layer=rckpt['config'].get('n_layer', 4), n_head=rckpt['config'].get('n_head', 4), n_embd=rckpt['config'].get('n_embd', 256)).to(device)
    rm.load_state_dict(rckpt['model'])
    rm.eval()

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # small prompt pool
    prompts = sample_prompts(16)

    step = 0
    while step < args.steps:
        # ----- COLLECT ROLLOUT BATCH -----
        batch_prompts = prompts[ (step*args.batch_size) % len(prompts) : ((step+1)*args.batch_size) % len(prompts) ]
        if len(batch_prompts) < args.batch_size:
            batch_prompts += prompts[:args.batch_size-len(batch_prompts)]
        texts = [format_prompt_only(p).replace("</s>", "") for p in batch_prompts]
        in_ids = [tok.encode(t) for t in texts]

        with torch.no_grad():
            out_ids = []
            for i, x in enumerate(in_ids):
                idx = torch.tensor([x], dtype=torch.long, device=device)
                out = policy.generate(idx, max_new_tokens=args.resp_len, temperature=0.2, top_k=3)
                out_ids.append(out[0].tolist())

        # split prompt/response per sample
        data = []
        for i, prompt in enumerate(batch_prompts):
            full = out_ids[i]
            # find boundary: index where prompt ends in the tokenized form
            # Use original prompt tokenization length (clipped by block_size)
            p_ids = in_ids[i][-block_size:]
            boundary = len(p_ids)
            resp_ids = full[boundary:]
            # compute rewards via RM on formatted prompt+response text
            resp_text = tok.decode(resp_ids)
            r_scalar = compute_reward(rm, tok, prompt, resp_text, device)
            data.append((torch.tensor(full, dtype=torch.long), boundary, r_scalar))

        # pad to same length
        policy_ctx = getattr(policy, "block_size", block_size)
        max_len = min(policy_ctx, max(t[0].numel() for t in data))
        B = len(data)
        seq = torch.zeros(B, max_len, dtype=torch.long, device=device)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        last_idx = torch.zeros(B, dtype=torch.long, device=device)
        rewards = torch.zeros(B, max_len, dtype=torch.float, device=device)

        for i, (ids, boundary, r_scalar) in enumerate(data):
            L_full = ids.numel()
            L = min(L_full, max_len)
            drop = L_full - L                 # tokens dropped from the left
            b = max(0, boundary - drop)       # shift boundary after left-trim
            seq[i, :L] = ids[-L:]
            if L < max_len:
                seq[i, L:] = 2  # fill remaining positions with <pad> token
            mask[i, b:L] = True
            rewards[i, L-1] = r_scalar
            last_idx[i] = L-1


        # logprobs & values for policy and reference
        # model_logprobs returns (B, T-1) for next-token logp; align to seq[:,1:]
        pol_lp = model_logprobs(policy, seq)
        ref_lp = model_logprobs(ref, seq)
        # values for seq positions (B,T)
        with torch.no_grad():
            logits, values, _ = policy(seq, None)
        values = values[:, :-1]  # align to pol_lp

        # Select only action positions
        act_mask = mask[:,1:]  # since logprobs are for predicting token t from <=t-1
        old_logp = pol_lp[act_mask].detach()
        ref_logp = ref_lp[act_mask].detach()
        old_values = values[act_mask].detach()

        # KL per action token and shaped rewards
        kl = (old_logp - ref_logp)
        shaped_r = rewards[:,1:][act_mask] - args.kl_coef * kl # penalty for drifting

        # Compute advantages/returns with last‑step bootstrap = 0 (episodic per response)
        # Flatten by sequence order inside each sample; we’ll approximate by grouping tokens per sample using last_idx.
        # For tutorial simplicity, treat advantages = shaped_r - old_values (no GAE). Works for end-only reward.
        returns = shaped_r  # target value = immediate shaped reward
        adv = returns - old_values
        # normalize adv
        adv = (adv - adv.mean()) / (adv.std().clamp_min(1e-6))

        # ----- UPDATE (single pass PPO for demo) -----
        # This step is done multiple times per batch in practice 
        policy.train()
        logits_new, values_new_full, _ = policy(seq, None)
        logp_full = torch.log_softmax(logits_new[:, :-1, :], dim=-1)
        labels = seq[:,1:]
        new_logp_all = logp_full.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        new_logp = new_logp_all[act_mask]
        new_values = values_new_full[:, :-1][act_mask]

        from ppo_loss import ppo_losses
        out_loss = ppo_losses(new_logp, old_logp, adv, new_values, old_values, returns,
                              clip_ratio=0.2, vf_coef=0.5, ent_coef=0.0)
        loss = out_loss.total_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        policy.eval()

        with torch.no_grad():
            # KL(old || new): movement of the updated policy from the snapshot used to collect data
            lp_post = model_logprobs(policy, seq)          # (B, T-1)
            lp_post = lp_post[act_mask]                    # only action positions
            kl_post = (old_logp - lp_post).mean()          # ≈ E[log π_old - log π_new]

            # KL(now || ref): how far the current policy is from the frozen reference
            lp_now = lp_post                               # already computed above on the same positions
            kl_ref_now = (lp_now - ref_logp).mean()        # ≈ E[log π_now - log π_ref]

        step += 1
        if step % 10 == 0:
            print(
                f"step {step} | loss {loss.item():.4f}"
                f"| value loss {out_loss.value_loss.item():.4f} | KL_move {kl_post.item():.6f} | KL_ref {kl_ref_now.item():.6f}"
            )


    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save({'model': policy.state_dict(), 'config': {
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
    }}, str(Path(args.out)/'model_last.pt'))
    print(f"Saved PPO policy to {args.out}/model_last.pt")

if __name__ == '__main__':
    main()