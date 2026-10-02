"""Timing and memory benchmarks. Every measurement runs in a fresh subprocess so that peak RSS is clean.

Single measurement (used by the stage driver):
  python -m msa.bench kernel --p 3 --impl custom --chunk 16 --B 8 --H 4 --T 128 --D 32 --L 4
  python -m msa.bench model  --task lm   --arm lp_3 --impl custom --chunk 16 --B 64 --threads 2
  python -m msa.bench model  --task mqar --arm lp_3 --P 64 --D 16 --B 128
Each prints one JSON line: sec_per_step, peak_rss_mb, base_rss_mb (RSS before the timed steps).
"""
import argparse
import json
import resource
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


def _rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 if sys.platform != "darwin" else r / 1024 / 1024   # Linux: KB, macOS: bytes


def bench_kernel(a):
    from . import lp
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    p = float(a.p)
    qs = [torch.randn(a.B, a.H, a.T, a.D, requires_grad=True) for _ in range(a.L)]
    ks = [torch.randn(a.B, a.H, a.T, a.D, requires_grad=True) for _ in range(a.L)]
    f = lambda q, k: lp.lp_dist(q, k, p, 1e-6, a.impl, a.chunk, a.causal)
    base = _rss_mb()

    def step():
        outs = [f(q, k) for q, k in zip(qs, ks)]
        sum(o.sum() for o in outs).backward()
    step()
    t0 = time.time()
    for _ in range(a.reps):
        step()
    return {"sec_per_step": (time.time() - t0) / a.reps, "peak_rss_mb": _rss_mb(), "base_rss_mb": base}


def bench_model(a):
    from . import train
    kw = dict(impl=a.impl, chunk=a.chunk, threads=a.threads, batch=a.B)
    if a.task == "mqar":
        kw.update(P=a.P, D=a.D)
    cfg = train.make_cfg(a.task, a.arm, **kw)
    torch.set_num_threads(a.threads)
    model = train.build_model(cfg)
    from . import calibration as cal
    cal.set_constants(model, train.get_constants(cfg))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    rng = np.random.default_rng(0)
    if a.task == "lm":
        data = train.get_lm_data(cfg["T"])
        mk = lambda: data.batch(rng, cfg["batch"])
    else:
        from . import mqar_gen
        def mk():
            c, q, t = mqar_gen.make_batch(rng, cfg["batch"], cfg["P"], cfg["Q"], cfg["n_keys"], cfg["n_vals"])
            return (torch.from_numpy(c), torch.from_numpy(q)), torch.from_numpy(t)
    def step():
        x, y = mk()
        out = model(*x) if a.task == "mqar" else model(x)
        loss = F.cross_entropy(out.reshape(-1, out.shape[-1]), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    base = _rss_mb()   # after model/data/calibration, before the first training step
    step()
    t0 = time.time()
    for _ in range(a.reps):
        step()
    return {"sec_per_step": (time.time() - t0) / a.reps, "peak_rss_mb": _rss_mb(), "base_rss_mb": base,
            "n_params": sum(p.numel() for p in model.parameters())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["kernel", "model"])
    ap.add_argument("--p", default="3"); ap.add_argument("--impl", default="auto")
    ap.add_argument("--chunk", type=int, default=16); ap.add_argument("--causal", type=int, default=0)
    ap.add_argument("--B", type=int, default=64); ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--T", type=int, default=128); ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--L", type=int, default=1); ap.add_argument("--P", type=int, default=16)
    ap.add_argument("--task", default="lm"); ap.add_argument("--arm", default="lp_3")
    ap.add_argument("--threads", type=int, default=2); ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args(argv)
    res = bench_kernel(a) if a.mode == "kernel" else bench_model(a)
    res["args"] = {k: v for k, v in vars(a).items() if k != "mode"}
    print("BENCH " + json.dumps(res))


class BudgetExceeded(Exception):
    """Raised when the wall-time budget for this invocation is used up (the bench stage is resumable)."""


_CACHE = {"path": None, "data": {}, "t0": time.time(), "budget": None}


def set_cache(path, budget=None, fresh=False):
    """Cache finished measurements in `path` so an interrupted bench stage can continue where it stopped."""
    import os
    _CACHE.update(path=str(path), t0=time.time(), budget=budget, data={})
    if os.path.exists(path) and not fresh:
        _CACHE["data"] = json.load(open(path))


def clear_cache():
    import os
    if _CACHE["path"] and os.path.exists(_CACHE["path"]):
        try:
            os.remove(_CACHE["path"])
        except OSError:                      # e.g. a mounted folder that forbids deletion: empty the cache instead
            json.dump({}, open(_CACHE["path"], "w"))
    _CACHE.update(path=None, data={}, budget=None)


def measure(*args, timeout=900):
    """Run one measurement in a subprocess; returns the parsed JSON or {'error': ...}."""
    key = " ".join(map(str, args))
    if key in _CACHE["data"]:
        return dict(_CACHE["data"][key])
    if _CACHE["budget"] is not None and time.time() - _CACHE["t0"] > _CACHE["budget"]:
        raise BudgetExceeded
    cmd = [sys.executable, "-m", "msa.bench", *map(str, args)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    res = {"error": (p.stderr or p.stdout)[-500:]}
    for line in p.stdout.splitlines():
        if line.startswith("BENCH "):
            res = json.loads(line[6:])
    if _CACHE["path"] and "error" not in res:
        _CACHE["data"][key] = res
        json.dump(_CACHE["data"], open(_CACHE["path"], "w"))
    return res


if __name__ == "__main__":
    main()
