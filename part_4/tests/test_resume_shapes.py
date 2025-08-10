import torch, tempfile, os
import torch.nn as nn
from checkpointing import save_checkpoint, load_checkpoint

class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.Linear(8,8)


def test_save_and_load(tmp_path):
    m = Dummy(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    class S: pass
    sch = S(); sch.__dict__ = {'warmup_steps': 10, 'total_steps': 100, 'base_lr': 1e-3, 'step_num': 5}
    class A: pass
    amp = A(); amp.scaler = torch.cuda.amp.GradScaler(enabled=False)

    out = tmp_path/"chk"
    save_checkpoint(m, opt, sch, amp, step=123, out_dir=str(out), tokenizer_dir=None)

    m2 = Dummy(); opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    sch2 = S(); sch2.__dict__ = {'warmup_steps': 1, 'total_steps': 1, 'base_lr': 1e-3, 'step_num': 0}
    amp2 = A(); amp2.scaler = torch.cuda.amp.GradScaler(enabled=False)

    step = load_checkpoint(m2, str(out/"model_last.pt"), optimizer=opt2, scheduler=sch2, amp=amp2)
    assert isinstance(step, int)