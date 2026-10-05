"""Evaluation and diagnostics (language-model loss; attention diagnostics shared by every task)."""
import math
import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ LM
@torch.no_grad()
def eval_lm(model, x, y, bs=256, max_windows=None):
    model.eval()
    if max_windows:
        x, y = x[:max_windows], y[:max_windows]
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        out = model(x[i:i + bs])
        tot += float(F.cross_entropy(out.reshape(-1, out.shape[-1]), y[i:i + bs].reshape(-1), reduction="sum"))
        n += y[i:i + bs].numel()
    loss = tot / n
    return {"loss": loss, "bpc": loss / math.log(2)}


# ------------------------------------------------------------------ attention diagnostics (any task)
def collect_scorer_stats(model, forward_fn, with_grad=None):
    """Run one forward (and optionally one backward) with collection on; return per-layer dicts."""
    scs = model.scorers()
    for sc in scs:
        sc.collect = True
        sc.collect_grad = with_grad is not None
        sc.grad_stats = None
    try:
        if with_grad is None:
            with torch.no_grad():
                forward_fn(model)
        else:
            model.zero_grad(set_to_none=True)
            loss = forward_fn(model)
            loss.backward()
            model.zero_grad(set_to_none=True)
    finally:
        for sc in scs:
            sc.collect = False
            sc.collect_grad = False
    out = []
    for sc in scs:
        d = dict(sc.stats)
        if sc.grad_stats:
            d.update({"grad_" + k: v for k, v in sc.grad_stats.items()})
        out.append(d)
    return out
