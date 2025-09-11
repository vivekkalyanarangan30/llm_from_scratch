import torch
from grpo_loss import ppo_policy_only_losses

def test_grpo_clipped_objective_behaves():
    N = 32
    old_logp = torch.zeros(N)                       # log π_old = 0 → π_old = 1.0
    new_logp = torch.log(torch.full((N,), 1.2))     # log π_new = log(1.2) → ratio=1.2
    adv      = torch.ones(N)                        # positive advantage
    kl_mean  = torch.tensor(0.5)                    # pretend KL(π||π_ref)=0.5

    out = ppo_policy_only_losses(
        new_logp=new_logp,
        old_logp=old_logp,
        adv=adv,
        clip_ratio=0.1,
        ent_coef=0.0,
        kl_coef=0.1,
        kl_mean=kl_mean,
    )

    # Ensure scalar loss
    assert out.total_loss.ndim == 0
    assert torch.isfinite(out.policy_loss)
    # KL penalty should have been added
    assert torch.allclose(out.kl_ref, kl_mean)
