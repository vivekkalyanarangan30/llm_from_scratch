from __future__ import annotations
import time

class NoopLogger:
    def log(self, **kwargs):
        pass
    def close(self):
        pass

class TBLogger(NoopLogger):
    """
    Backward compatible:
      - logger.log(step=..., loss=..., lr=...)
    Extras you can optionally use:
      - logger.hist("params/wte.weight", tensor, step)
      - logger.text("samples/generation", text, step)
      - logger.image("attn/heatmap", HWC_or_CHW_tensor_or_np, step)
      - logger.graph(model, example_batch)
      - logger.hparams(dict_of_config, dict_of_metrics_once)
      - logger.flush()
    Auto-behavior:
      - If a value in .log(...) is a tensor/ndarray with >1 element, it logs a histogram.
      - If key starts with "text/", logs as text.
    """
    def __init__(self, out_dir: str, flush_secs: int = 10):
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            self._tb_ok = True
        except Exception:
            self._tb_ok = False
            self.w = None
            return
        from torch.utils.tensorboard import SummaryWriter  # re-import in scope

        self.w = SummaryWriter(log_dir=out_dir, flush_secs=flush_secs)

        # simple heuristics
        self._auto_hist_max_elems = 2048  # won't histogram giant tensors by accident

    # ---------- backwards-compatible ----------
    def log(self, step: Optional[int] = None, **kv: Any):
        if not self.w: return
        for k, v in kv.items():
            # text channel (opt-in via key prefix)
            if isinstance(k, str) and k.startswith("text/"):
                try:
                    self.w.add_text(k[5:], str(v), global_step=step)
                except Exception:
                    pass
                continue

            # scalar vs histogram auto-route
            try:
                import torch, numpy as np  # lazy
                is_torch = isinstance(v, torch.Tensor)
                is_np = isinstance(v, np.ndarray)
                if is_torch or is_np:
                    # scalar?
                    numel = int(v.numel() if is_torch else v.size)
                    if numel == 1:
                        val = (v.item() if is_torch else float(v))
                        self.w.add_scalar(k, float(val), global_step=step)
                    else:
                        # small-ish tensors => histogram
                        if numel <= self._auto_hist_max_elems:
                            self.w.add_histogram(k, v.detach().cpu() if is_torch else v, global_step=step)
                        else:
                            # fall back to scalar summary stats
                            arr = v.detach().cpu().flatten().numpy() if is_torch else v.flatten()
                            self.w.add_scalar(k + "/mean", float(arr.mean()), global_step=step)
                            self.w.add_scalar(k + "/std", float(arr.std()), global_step=step)
                    continue
            except Exception:
                pass

            # number-like
            try:
                self.w.add_scalar(k, float(v), global_step=step)
            except Exception:
                # swallow non-numeric junk silently (same behavior as before)
                pass

    # ---------- nice-to-have helpers ----------
    def hist(self, tag: str, values: Any, step: Optional[int] = None, bins: str = "tensorflow"):
        if not self.w: return
        try:
            import torch
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu()
            self.w.add_histogram(tag, values, global_step=step, bins=bins)
        except Exception:
            pass

    def text(self, tag: str, text: str, step: Optional[int] = None):
        if not self.w: return
        try:
            self.w.add_text(tag, text, global_step=step)
        except Exception:
            pass

    def image(self, tag: str, img, step: Optional[int] = None):
        """
        img: torch.Tensor [C,H,W] or [H,W,C] or numpy array
        """
        if not self.w: return
        try:
            self.w.add_image(tag, img, global_step=step, dataformats="CHW" if getattr(img, "ndim", 0) == 3 and img.shape[0] in (1,3) else "HWC")
        except Exception:
            pass

    def graph(self, model, example_input):
        if not self.w: return
        try:
            # example_input: a Tensor batch or a tuple
            if not isinstance(example_input, tuple):
                example_input = (example_input,)
            self.w.add_graph(model, example_input)
        except Exception:
            pass  # graph tracing can fail depending on model control flow; don't crash

    def hparams(self, hparams: Dict[str, Any], metrics_once: Optional[Dict[str, float]] = None):
        if not self.w: return
        try:
            self.w.add_hparams(hparams, metrics_once or {})
        except Exception:
            pass

    def flush(self):
        if self.w:
            try: self.w.flush()
            except Exception: pass

    def close(self):
        if self.w:
            try: self.w.close()
            except Exception: pass

class WBLogger(NoopLogger):
    def __init__(self, project: str, run_name: str | None = None):
        try:
            import wandb
            wandb.init(project=project, name=run_name)
            self.wb = wandb
        except Exception:
            self.wb = None
    def log(self, **kv):
        if self.wb: self.wb.log(kv)


def init_logger(which: str, out_dir: str = "runs/part4"):
    if which == 'tensorboard':
        return TBLogger(out_dir)
    if which == 'wandb':
        return WBLogger(project='llm-part4')
    return NoopLogger()