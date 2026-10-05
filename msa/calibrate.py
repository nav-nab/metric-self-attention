"""The calibration constants of design section 6.

For every arm, head size and layer, the scaled scores of a query have a standard
deviation of 1 across its allowed keys at initialisation. The measurement uses a
model built with the calibration seed, run on fixed batches, layer by layer.

Pooling (implementation plan section 5): the variance of the raw score is computed for
each query over its allowed keys. These variances are averaged over the queries with at
least MIN_KEYS allowed keys, over the four heads of the layer, and over the batches.
The constant is c = 1 / sqrt(mean variance).

For the distance arms with 1 < p < infinity the typical sum S_typ is the median of
sum_m |x_m - y_m|^p over the same pairs of query and key. The epsilon is r * S_typ.
"""
import csv
import io
import math
from pathlib import Path

import torch

from .data import CALIBRATION_SEED, calibration_batches, load_corpus
from .model import build_model
from .scores import CROSSED_ARMS, EPS_RATIO, HEAD_SIZES, INF, parse_arm
from .util import atomic_write_json, atomic_write_text

MIN_KEYS = 8                 # queries with fewer allowed keys are excluded
N_CALIBRATION_BATCHES = 32   # batches of 64 sequences; halves agree to well under 1 percent


def calibration_combos():
    """The (arm, head size) pairs that need constants: 45 crossed cells and the unmasked arm."""
    combos = [(arm, D) for arm in CROSSED_ARMS for D in HEAD_SIZES]
    combos.append(("dot_untied_unmasked", 16))
    return combos


def calibration_path(root, arm, D):
    return Path(root) / "calibration" / f"{arm}_D{D}.json"


def load_calibration(root, arm, D):
    """The per-layer constants for one cell: a list of dicts with the keys "c" and "eps"."""
    path = calibration_path(root, arm, D)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run the calibrate stage first.")
    import json
    return json.loads(path.read_text())["layers"]


def _collect(attn, raw, min_keys):
    """Per-query variances of the raw scores and, for distances, the sums |x - y|^p."""
    T = raw.shape[-1]
    allowed = attn.allowed[:T, :T]
    n = allowed.sum(-1)
    rows = n >= min_keys
    mask = allowed[rows]                                  # (R, T)
    cnt = n[rows].to(raw.dtype)                           # (R,)
    r = raw[:, :, rows, :]                                # (B, H, R, T)
    m = mask.to(raw.dtype)
    mean = (r * m).sum(-1) / cnt
    var = (((r - mean.unsqueeze(-1)) * m) ** 2).sum(-1) / cnt
    sums = None
    p = attn.spec.p
    if attn.spec.kind == "distance" and p not in (1.0, INF):
        sums = (-r[:, :, mask]).pow(p).flatten()
    return var.double().sum().item(), var.numel(), sums


@torch.no_grad()
def calibrate_model(model, batches, r=EPS_RATIO, min_keys=MIN_KEYS):
    """Set c and eps in every layer of the model and return the measurements."""
    spec = parse_arm(model.cfg.arm)
    if not spec.has_scores:
        raise ValueError("the uniform arm has no scores to calibrate")
    model.eval()
    results = []
    for layer, block in enumerate(model.blocks):
        attn = block.attn
        attn.c.fill_(1.0)
        attn.eps.fill_(0.0)          # S_typ is then exactly dist^p
        attn.capture = True
        per_batch, sums, z_sq, z_n = [], [], 0.0, 0
        for x, _ in batches:
            model.hidden(x, upto=layer + 1)
            cap = attn.captured
            var_sum, count, s = _collect(attn, cap["raw"], min_keys)
            per_batch.append((var_sum, count))
            if s is not None:
                sums.append(s)
            z_sq += cap["q"].double().pow(2).sum().item()
            z_n += cap["q"].numel()
        attn.capture = False
        attn.captured = None
        total_var = sum(v for v, _ in per_batch)
        total_n = sum(n for _, n in per_batch)

        def sigma_of(items):
            return math.sqrt(sum(v for v, _ in items) / sum(n for _, n in items))

        sigma = math.sqrt(total_var / total_n)
        s_typ = torch.cat(sums).median().item() if sums else None
        eps = r * s_typ if s_typ is not None else 0.0
        attn.c.fill_(1.0 / sigma)
        attn.eps.fill_(eps)
        results.append({
            "layer": layer, "c": 1.0 / sigma, "sigma": sigma, "s_typ": s_typ, "eps": eps,
            "n_queries": total_n,
            "sigma_halves": [sigma_of(per_batch[0::2]), sigma_of(per_batch[1::2])],
            "z_var": z_sq / z_n,
        })
    return results


def run_calibration(root, n_batches=N_CALIBRATION_BATCHES, combos=None, log=print):
    """Calibrate every combination that has no file yet. Finished combinations are skipped."""
    corpus = load_corpus()
    batches = calibration_batches(corpus, n_batches)
    combos = combos if combos is not None else calibration_combos()
    for arm, D in combos:
        path = calibration_path(root, arm, D)
        if path.exists():
            log(f"calibration: {arm} D={D} is done, skipped")
            continue
        model = build_model(arm, D, CALIBRATION_SEED, corpus.vocab_size)
        layers = calibrate_model(model, batches)
        atomic_write_json(path, {"arm": arm, "D": D, "seed": CALIBRATION_SEED,
                                 "n_batches": n_batches, "r": EPS_RATIO, "min_keys": MIN_KEYS,
                                 "layers": layers})
        log(f"calibration: {arm} D={D} c=" + ", ".join(f"{x['c']:.4g}" for x in layers))
    write_summary(root, combos)


def write_summary(root, combos=None):
    """One table of all constants, for the reader. It is rebuilt from the per-cell files."""
    import json
    combos = combos if combos is not None else calibration_combos()
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["arm", "D", "layer", "c", "sigma", "s_typ", "eps", "z_var", "c_zvar_sqrtD"])
    for arm, D in combos:
        path = calibration_path(root, arm, D)
        if not path.exists():
            continue
        for x in json.loads(path.read_text())["layers"]:
            check = x["c"] * x["z_var"] * math.sqrt(D) if parse_arm(arm).kind == "dot" else ""
            w.writerow([arm, D, x["layer"], x["c"], x["sigma"], x["s_typ"] or "", x["eps"],
                        x["z_var"], check])
    atomic_write_text(Path(root) / "calibration" / "calibration_summary.csv", out.getvalue())
