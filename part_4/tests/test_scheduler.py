from lr_scheduler import WarmupCosineLR

class DummyOpt:
    def __init__(self):
        self.param_groups = [{'lr': 0.0}]

def test_warmup_cosine_lr_progression():
    opt = DummyOpt()
    sch = WarmupCosineLR(opt, warmup_steps=10, total_steps=110, base_lr=1e-3)
    lrs = [sch.step() for _ in range(110)]
    assert max(lrs) <= 1e-3 + 1e-12
    assert lrs[0] < lrs[9]  # warmup increasing
    assert lrs[-1] < lrs[10]  # cosine decays