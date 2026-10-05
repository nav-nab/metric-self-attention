"""Empirical calibration of the logit scale constant c and the relative epsilon.

For each layer (in order) and each arm, at initialisation (calibration seed, same for all arms):
  * measure the raw score spread: for each query, the standard deviation of the raw score
    (distance, or dot product) over its allowed keys; average over queries (with at least `min_keys`
    allowed keys), heads and batches  ->  std_raw.
  * c = target / std_raw, so that the logit spread at initialisation (beta = 1) equals `target`.
  * measure the typical sum inside the root S_typ = mean over allowed pairs of sum_d |dz_d|^p
    (finite p only); eps = r * S_typ.
Layer l is calibrated with layers < l already calibrated, so its inputs match the calibrated model.
"""
import numpy as np
import torch

CALIB_SEED = 12345


@torch.no_grad()
def calibrate_model(model, forward_fns, target=1.0, r=1e-3):
    """forward_fns: list of callables f(model) running one calibration batch. Returns per-layer dicts."""
    model.eval()
    scs = model.scorers()
    out = []
    for l, sc in enumerate(scs):
        sc.calib = True
        sc.c.fill_(1.0)
        sc.eps.fill_(0.0)
        acc = None
        for f in forward_fns:
            f(model)
            o = sc.calib_out
            if acc is None:
                acc = {k: ([x for x in v] if isinstance(v, list) else v) for k, v in o.items()}
                acc["n"] = 1
                acc["S_medians"] = [o.get("S_median")]
            else:
                for k in ("std_sum", "std_count", "S_sum", "S_count", "std_per_head_count"):
                    if k in o:
                        acc[k] += o[k]
                acc["std_per_head_sum"] = [a + b for a, b in zip(acc["std_per_head_sum"], o["std_per_head_sum"])]
                acc["S_medians"].append(o.get("S_median"))
                acc["n"] += 1
        sc.calib = False
        std_raw = acc["std_sum"] / acc["std_count"]
        per_head = [s / acc["std_per_head_count"] for s in acc["std_per_head_sum"]]
        rec = {"layer": l, "arm": sc.arm, "std_raw": std_raw, "std_raw_per_head": per_head,
               "c": target / std_raw, "target": target}
        if "S_sum" in acc:
            S_typ = acc["S_sum"] / acc["S_count"]
            rec.update({"S_typ": S_typ, "S_median": float(np.mean(acc["S_medians"])),
                        "S_q01": acc["S_q01"], "S_q99": acc["S_q99"], "r": r, "eps": r * S_typ})
        sc.c.fill_(rec["c"])
        sc.eps.fill_(rec.get("eps", 0.0))
        out.append(rec)
    return out


def set_constants(model, consts):
    for sc, rec in zip(model.scorers(), consts):
        sc.c.fill_(rec["c"])
        sc.eps.fill_(rec.get("eps", 0.0))
