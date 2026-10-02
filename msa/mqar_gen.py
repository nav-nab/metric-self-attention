"""MQAR variant A data generator.

Context: P distinct keys with random values (values drawn with replacement).
Queries: Q of the P context keys, drawn without replacement, in random order.
Targets: the value paired with each queried key.
The reference implementation (per-sequence loop, from the design stage) is kept as
`make_batch_reference`; `make_batch` is the vectorised version used for training.
Both are checked in `check_generator`.
"""
import numpy as np

BLANK = -1  # id of the "no value" token used in query tokens (mapped to row n_vals of the value table)


def make_batch_reference(rng, n, P, Q, n_keys, n_vals):
    ctx = np.zeros((n, P, 2), dtype=np.int64)
    qry = np.zeros((n, Q), dtype=np.int64)
    tgt = np.zeros((n, Q), dtype=np.int64)
    for b in range(n):
        keys = rng.choice(n_keys, size=P, replace=False)
        vals = rng.integers(0, n_vals, size=P)
        ctx[b, :, 0], ctx[b, :, 1] = keys, vals
        pick = rng.choice(P, size=Q, replace=False)
        qry[b] = keys[pick]
        tgt[b] = vals[pick]
    return ctx, qry, tgt


def make_batch(rng, n, P, Q, n_keys, n_vals):
    """Vectorised variant A. Returns ctx (n,P,2) [key,val], qry (n,Q) key ids, tgt (n,Q)."""
    assert P <= n_keys and Q <= P
    keys = np.argsort(rng.random((n, n_keys)), axis=1)[:, :P]            # distinct keys
    vals = rng.integers(0, n_vals, size=(n, P))                          # values with replacement
    pick = np.argsort(rng.random((n, P)), axis=1)[:, :Q]                 # queried positions, random order
    ctx = np.stack([keys, vals], axis=-1).astype(np.int64)
    qry = np.take_along_axis(keys, pick, axis=1).astype(np.int64)
    tgt = np.take_along_axis(vals, pick, axis=1).astype(np.int64)
    return ctx, qry, tgt


def fixed_set(seed, n, P, Q, n_keys, n_vals):
    """Fixed evaluation set (identical for every arm): depends only on the seed and the parameters."""
    rng = np.random.default_rng(seed)
    return make_batch(rng, n, P, Q, n_keys, n_vals)


def baselines(P, n_vals):
    """No-context chance and key-blind guess (a random context value)."""
    return {"chance": 1.0 / n_vals, "key_blind": 1.0 / P + (1.0 - 1.0 / P) / n_vals}


def fmt(ctx, qry, tgt):
    s = " ".join(f"[{k},{v}]" for k, v in ctx)
    q = " ".join(f"[{k},_]" for k in qry)
    return f"context: {s}\nqueries: {q}\ntargets: {[int(t) for t in tgt]}"


def check_generator(P=16, Q=8, n_keys=256, n_vals=64, n=20000, seed=0, check_n=2000):
    """Property checks and empirical baselines. Returns a dict of results (all booleans must be True)."""
    rng = np.random.default_rng(seed)
    out = {"P": P, "Q": Q, "n_keys": n_keys, "n_vals": n_vals}
    for name, fn in [("vectorised", make_batch), ("reference", make_batch_reference)]:
        ctx, qry, tgt = fn(rng, n, P, Q, n_keys, n_vals)
        r = {}
        r["keys_distinct"] = all(len(set(c[:, 0])) == P for c in ctx[:check_n])
        r["queries_from_context"] = all(set(qry[b]).issubset(set(ctx[b, :, 0])) for b in range(check_n))
        r["each_key_queried_once"] = all(len(set(qry[b])) == Q for b in range(check_n))
        r["answers_consistent"] = all(dict(zip(ctx[b, :, 0], ctx[b, :, 1]))[qry[b, j]] == tgt[b, j]
                                      for b in range(check_n) for j in range(Q))
        r["ranges_ok"] = bool(ctx[:, :, 0].max() < n_keys and ctx[:, :, 1].max() < n_vals and ctx.min() >= 0)
        # statistical checks
        r["val_uniform_chi2_p"] = _chi2_uniform(ctx[:, :, 1].ravel(), n_vals)
        r["key_uniform_chi2_p"] = _chi2_uniform(ctx[:, :, 0].ravel(), n_keys)
        r["queried_position_uniform_chi2_p"] = _queried_pos_uniform(ctx, qry, P)
        b = baselines(P, n_vals)
        acc_noctx = float(np.mean(rng.integers(0, n_vals, size=tgt.shape) == tgt))
        guess = np.take_along_axis(ctx[:, :, 1], rng.integers(0, P, size=(len(ctx), Q)), axis=1)
        acc_blind = float(np.mean(guess == tgt))
        r["chance_theory"], r["chance_sim"] = b["chance"], acc_noctx
        r["key_blind_theory"], r["key_blind_sim"] = b["key_blind"], acc_blind
        r["baselines_match"] = bool(abs(acc_noctx - b["chance"]) < 0.004 and abs(acc_blind - b["key_blind"]) < 0.006)
        out[name] = r
    out["ok"] = all(v for d in (out["vectorised"], out["reference"]) for k, v in d.items() if isinstance(v, bool))
    # statistical tests: p > 1e-3 required
    for d in (out["vectorised"], out["reference"]):
        for k in list(d):
            if k.endswith("_chi2_p"):
                out["ok"] &= bool(d[k] > 1e-3)
    return out


def _chi2_uniform(x, k):
    from scipy.stats import chisquare
    counts = np.bincount(x, minlength=k)
    return float(chisquare(counts).pvalue)


def _queried_pos_uniform(ctx, qry, P):
    """Each context position should be equally likely to be the queried one (random order, no bias)."""
    from scipy.stats import chisquare
    pos = np.array([[int(np.where(ctx[b, :, 0] == k)[0][0]) for k in qry[b]] for b in range(2000)]).ravel()
    return float(chisquare(np.bincount(pos, minlength=P)).pvalue)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ctx, qry, tgt = make_batch(rng, 1, P=6, Q=3, n_keys=32, n_vals=16)
    print(fmt(ctx[0], qry[0], tgt[0]))
    import json
    print(json.dumps(check_generator(), indent=1))
