"""Evaluation and diagnostics for both tasks."""
import math
import numpy as np
import torch
import torch.nn.functional as F


# ------------------------------------------------------------------ MQAR
def _tt(a):
    return torch.from_numpy(a)


@torch.no_grad()
def eval_mqar(model, ctx, qry, tgt, bs=1000, diag=False, ablate_context=False):
    model.eval()
    n = len(ctx)
    correct = 0
    total = 0
    loss_sum = 0.0
    d = {"top1_att": 0.0, "att_mass_correct": 0.0, "entropy": 0.0, "margin_logit": 0.0, "margin_dist": 0.0,
         "hub_corr": 0.0, "n_hub": 0}
    for i in range(0, n, bs):
        c, q, t = _tt(ctx[i:i + bs]), _tt(qry[i:i + bs]), _tt(tgt[i:i + bs])
        if diag:
            out, att, logits, zk = model(c, q, ablate_context=ablate_context, return_att=True)
        else:
            out = model(c, q, ablate_context=ablate_context)
        correct += int((out.argmax(-1) == t).sum())
        total += t.numel()
        loss_sum += float(F.cross_entropy(out.reshape(-1, out.shape[-1]), t.reshape(-1), reduction="sum"))
        if diag:
            B, Q, P = att.shape
            match = (c[:, None, :, 0] == q[:, :, None])                       # B,Q,P  (exactly one True per row)
            pos = match.float().argmax(-1)                                    # B,Q
            d["top1_att"] += float((att.argmax(-1) == pos).float().sum())
            d["att_mass_correct"] += float((att * match.float()).sum())
            d["entropy"] += float((-(att * att.clamp_min(1e-30).log()).sum(-1)).sum())
            lc = (logits * match.float()).sum(-1)
            lr = logits.masked_fill(match, -1e30).amax(-1)
            d["margin_logit"] += float((lc - lr).sum())
            if model.scorer.kind == "lp":
                dist = -logits / (model.scorer.log_beta.exp() * model.scorer.c)
                dc = (dist * match.float()).sum(-1)
                dr = dist.masked_fill(match, 1e30).amin(-1)
                d["margin_dist"] += float((dr - dc).sum())
            # hub measure: per sequence, correlation across context keys of key norm and attention received
            recv = att.sum(1)                                                 # B,P
            kn = zk.norm(dim=-1)                                              # B,P
            a, b = kn - kn.mean(1, keepdim=True), recv - recv.mean(1, keepdim=True)
            den = a.norm(dim=1) * b.norm(dim=1)
            ok = den > 1e-12
            if ok.any():
                d["hub_corr"] += float(((a * b).sum(1)[ok] / den[ok]).sum())
                d["n_hub"] += int(ok.sum())
    res = {"acc": correct / total, "loss": loss_sum / total}
    if diag:
        nq = total
        res.update({"top1_att": d["top1_att"] / nq, "att_mass_correct": d["att_mass_correct"] / nq,
                    "entropy": d["entropy"] / nq, "margin_logit": d["margin_logit"] / nq,
                    "hub_corr": d["hub_corr"] / max(d["n_hub"], 1)})
        if model.scorer.kind == "lp":
            res["margin_dist"] = d["margin_dist"] / nq
    return res


def shuffle_context(ctx, seed=0):
    rng = np.random.default_rng(seed)
    perm = np.argsort(rng.random(ctx.shape[:2]), axis=1)
    return np.take_along_axis(ctx, perm[:, :, None], axis=1)


def mqar_sanity(model, ctx, qry, tgt):
    """Shuffling the context must leave accuracy unchanged; hiding the context must give chance."""
    base = eval_mqar(model, ctx, qry, tgt)
    shuf = eval_mqar(model, shuffle_context(ctx), qry, tgt)
    nocx = eval_mqar(model, ctx, qry, tgt, ablate_context=True)
    return {"acc": base["acc"], "acc_shuffled": shuf["acc"], "acc_no_context": nocx["acc"],
            "shuffle_abs_diff": abs(base["acc"] - shuf["acc"]), "chance": 1.0 / model.n_vals}


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


# ------------------------------------------------------------------ attention diagnostics (both tasks)
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
