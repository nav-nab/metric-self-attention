"""Pilot stages for the language model. Each stage writes results/pilot/<stage>.json (and figures) and returns its result dict.
Called from execute_project.py; every stage can also be re-run on its own.

The recall (MQAR) stages of the original pilot live in mqar/ (see the README there). The generic helpers at the
top of this file (_save, _load, hdr, _rows, _summarise_trial, mde_in_sd_units, paired_summary) are shared with the shelf."""
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from . import bench as B
from . import gradcheck as GC
from . import lp, runner, train
from .arms import ARMS, NULL_ARM

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "results" / "pilot"
LM_DS = (16, 12, 8, 4, 2)  # head sizes (query-key dimension per head) used by the calibration and timing stages
def _save(name, obj, out=None):
    out = Path(out or PILOT)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(out / f"{name}.json", "w"), indent=1, default=float)
    return obj


def _load(name, out=None):
    p = Path(out or PILOT) / f"{name}.json"
    return json.load(open(p)) if p.exists() else None


def hdr(s):
    print("\n" + "=" * 8 + " " + s + " " + "=" * 8, flush=True)


# ------------------------------------------------------------------ 1. environment
def stage_env(out=None, **_):
    hdr("environment")
    import numpy
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "torch": torch.__version__, "numpy": numpy.__version__,
            "torch_threads_default": torch.get_num_threads(), "cpu_count": os.cpu_count(),
            "mkl": torch.backends.mkl.is_available(), "mkldnn": torch.backends.mkldnn.is_available()}
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except Exception:  # noqa
        info["scipy"] = None
    warns = []
    pv = sys.version_info
    if not ((3, 10) <= (pv.major, pv.minor) <= (3, 12)):
        warns.append("Python outside 3.10-3.12 (torch 2.2.2 has no wheels for it)")
    if not torch.__version__.startswith("2.2.2"):
        warns.append(f"torch {torch.__version__} differs from the pinned 2.2.2 (numbers are not comparable with laptop runs)")
    if int(numpy.__version__.split(".")[0]) >= 2:
        warns.append("numpy >= 2 with torch 2.2.2 can fail; requirements pin numpy<2")
    info["warnings"] = warns
    # smoke: two training steps for each arm family, at head size 16 and at a small head size, plus the uniform control
    t0 = time.time()
    for arm, qk in (("dot_tied", None), ("lp_3", None), ("lp_inf", None), ("lp_2", 2), (NULL_ARM, None)):
        extra = {"qk_dim": qk} if qk else {}
        cfg = train.make_cfg("lm", arm, steps=2, eval_every=2, diag_every=2, ckpt_every=100, log_every=1, threads=1,
                             eval_windows_mid=8, **extra)
        d = tempfile.mkdtemp()
        r = train.run(cfg, d, verbose=False)
        assert not r["diverged"]
        shutil.rmtree(d)
    info["smoke_train_ok"] = True
    info["smoke_seconds"] = time.time() - t0
    print(json.dumps(info, indent=1))
    return _save("env", info, out)


# ------------------------------------------------------------------ 2. gradient check
def stage_gradcheck(out=None, **_):
    hdr("gradient check (float64, custom vs naive)")
    r = GC.run_gradcheck(verbose=False)
    worst = max(max(x["max_abs_err_out_dq_dk"]) for x in r["float64_vs_naive"])
    print(f"all pass: {r['ok']}; worst float64 abs error over out/dq/dk: {worst:.2e}")
    r["worst_float64_abs_err"] = worst
    return _save("gradcheck", r, out)


# ------------------------------------------------------------------ 3. timing and memory
def _fmt_bench(r):
    if "error" in r:
        return "ERR"
    return f"{r['sec_per_step']:.3f}s/{r['peak_rss_mb']:.0f}MB"


def stage_bench(out=None, quick=False, threads=2, time_budget=None, fresh=False, **_):
    """Timing/memory per arm at the language-model size (and at small head sizes), plus implementation/chunk comparisons.
    Resumable: finished measurements are cached; with time_budget (seconds) a call stops early and the next call continues."""
    hdr(f"timing and memory (threads={threads})")
    Path(out or PILOT).mkdir(parents=True, exist_ok=True)
    B.set_cache(Path(out or PILOT) / ".bench_cache.json", time_budget, fresh)
    try:
        return _stage_bench(out, quick, threads)
    except B.BudgetExceeded:
        print("[bench] time budget used up; run the same command again to continue")
        return None


def _stage_bench(out, quick, threads):
    reps = 3 if quick else 5
    res = {"threads": threads, "kernel_ref": [], "chunk_sweep": [], "chunk_sweep_model": [], "lm_arms": [], "lm_impls": [],
           "lm_small_d": [], "threads_cmp": [], "parallel": {}}
    # (a) kernel reference setting from the design stage: B=8,H=4,T=128,D=32, 4 layers alive, chunk 16
    for p in (["3"] if quick else ["1", "1.5", "2", "3", "8", "inf"]):
        impls = ["naive", "ckpt", "custom"] + (["matmul"] if p == "2" else [])
        for impl in impls:
            r = B.measure("kernel", "--p", p, "--impl", impl, "--chunk", 16, "--B", 8, "--H", 4, "--T", 128, "--D", 32,
                          "--L", 4, "--threads", threads, "--reps", reps)
            r.update(p=p, impl=impl)
            res["kernel_ref"].append(r)
            print(f"  kernel ref p={p:>3} {impl:7s}: {_fmt_bench(r)}", flush=True)
    # (b) chunk sweep at the language-model attention size (B=64,H=4,T=64,D=16, 2 layers alive), causal block skipping on/off
    for p in (["3"] if quick else ["3", "inf"]):
        for chunk in ([16] if quick else [4, 8, 16, 32, 64]):
            for causal in (1, 0):
                r = B.measure("kernel", "--p", p, "--impl", "custom", "--chunk", chunk, "--causal", causal, "--B", 64,
                              "--H", 4, "--T", 64, "--D", 16, "--L", 2, "--threads", threads, "--reps", reps)
                r.update(p=p, chunk=chunk, causal=causal)
                res["chunk_sweep"].append(r)
                print(f"  chunk sweep p={p:>3} chunk={chunk:<3} causal={causal}: {_fmt_bench(r)}", flush=True)
    # (b2) the same sweep at model level (full training step, 1 thread, more repetitions): this decided the default chunk
    for arm in (["lp_3"] if quick else ["lp_1.5", "lp_3", "lp_inf"]):
        for chunk in ([4, 16] if quick else [2, 4, 8, 16, 32]):
            r = B.measure("model", "--arm", arm, "--chunk", chunk, "--B", 64, "--threads", 1, "--reps", 4 if quick else 12)
            r.update(arm=arm, chunk=chunk)
            res["chunk_sweep_model"].append(r)
            print(f"  model chunk sweep {arm:7s} chunk={chunk:<3}: {_fmt_bench(r)}", flush=True)
    # (c) full training step per arm (B=64, T=64, 2 layers, 4 heads of 16)
    for arm in (["dot_tied", "lp_3"] if quick else ARMS):
        r = B.measure("model", "--arm", arm, "--B", 64, "--threads", threads, "--reps", reps)
        r.update(arm=arm)
        res["lm_arms"].append(r)
        print(f"  LM step arm={arm:10s}: {_fmt_bench(r)}  (params {r.get('n_params')})", flush=True)
    for impl in ([] if quick else ["naive", "ckpt", "custom"]):
        for arm in ("lp_3", "lp_inf"):
            r = B.measure("model", "--arm", arm, "--impl", impl, "--B", 64, "--threads", threads, "--reps", reps)
            r.update(arm=arm, impl=impl)
            res["lm_impls"].append(r)
            print(f"  LM step arm={arm:7s} impl={impl:6s}: {_fmt_bench(r)}", flush=True)
    # (d) small head sizes: the query-key dimension shrinks, so the cost of a distance step should fall
    for qk in ([4] if quick else [8, 4, 2]):
        for arm in (["dot_tied", "lp_3"] if quick else ARMS + [NULL_ARM]):
            r = B.measure("model", "--arm", arm, "--qk-dim", qk, "--B", 64, "--threads", threads, "--reps", reps)
            r.update(arm=arm, qk_dim=qk)
            res["lm_small_d"].append(r)
            print(f"  LM step D={qk} arm={arm:10s}: {_fmt_bench(r)}", flush=True)
    # (e) threads and process-level parallelism (lp_3 and dot_tied)
    if not quick and (os.cpu_count() or 1) >= 2:
        import subprocess
        for arm in ("lp_3", "dot_tied"):
            for th in sorted({1, threads}):
                r = B.measure("model", "--arm", arm, "--B", 64, "--threads", th, "--reps", reps)
                r.update(arm=arm, threads=th)
                res["threads_cmp"].append(r)
                print(f"  LM {arm} threads={th}: {_fmt_bench(r)}", flush=True)
        # two concurrent single-thread processes vs one process with all threads
        n = min(threads, os.cpu_count() or 1)
        cmd = [sys.executable, "-m", "msa.bench", "model", "--arm", "lp_3", "--B", "64", "--threads", "1", "--reps", str(reps)]
        import subprocess as sp
        procs = [sp.Popen(cmd, stdout=sp.PIPE, text=True, cwd=ROOT) for _ in range(n)]
        outs = []
        for pr in procs:
            o = pr.communicate()[0]
            outs += [json.loads(l[6:]) for l in o.splitlines() if l.startswith("BENCH ")]
        res["parallel"] = {"n_procs": n, "per_proc_sec_per_step": [o["sec_per_step"] for o in outs]}
        single = [r for r in res["threads_cmp"] if r["arm"] == "lp_3" and r["threads"] == n]
        if single and outs:
            tput_conc = n / np.mean([o["sec_per_step"] for o in outs])
            tput_single = 1 / single[0]["sec_per_step"]
            res["parallel"].update(throughput_concurrent_steps_per_s=tput_conc, throughput_single_proc_steps_per_s=tput_single,
                                   concurrent_over_single=tput_conc / tput_single)
            print(f"  {n} concurrent 1-thread procs: {tput_conc:.2f} steps/s total vs 1 proc x {n} threads: {tput_single:.2f} steps/s", flush=True)
    B.clear_cache()
    return _save("bench" + ("_quick" if quick else ""), res, out)


# ------------------------------------------------------------------ 4. calibration
def stage_calibrate(out=None, target=1.0, r=None, Ds=None, **_):
    """Calibration constants c, typical sums inside the root, and the epsilon implied by r, for each head size D.
    The constants are recomputed for every head size (decided 2026-10-05). r = 1e-12 is proposed, not yet approved."""
    hdr("calibration at initialisation")
    r = train.DEFAULTS["common"]["r"] if r is None else r
    Ds = list(Ds or LM_DS)
    res = {"target": target, "r": r, "Ds": Ds, "lm": {}}
    for D in Ds:
        res["lm"][str(D)] = {arm: train.get_constants(train.make_cfg("lm", arm, target=target, r=r, qk_dim=D)) for arm in ARMS}
    # sanity: the tied dot product constant should be near 1/sqrt(D)
    res["dot_sanity"] = {str(D): {"c_layers": [l["c"] for l in res["lm"][str(D)]["dot_tied"]], "one_over_sqrt_D": 1 / math.sqrt(D)} for D in Ds}
    print(f"target logit spread = {target}, provisional r = {r}")
    for D in Ds:
        print(f"--- head size D = {D}: arm, c, S_typ (mean sum inside the root), S_median, eps ---")
        for arm in ARMS:
            for l in res["lm"][str(D)][arm]:
                print(f"  {arm:10s} L{l['layer']}: c={l['c']:.4f} std_raw={l['std_raw']:.4f}"
                      + (f" S_typ={l['S_typ']:.4g} S_med={l['S_median']:.4g} eps={l['eps']:.3g}" if "S_typ" in l else ""))
    print("dot_tied sanity c vs 1/sqrt(D):", json.dumps(res["dot_sanity"]))
    return _save("calibrate", res, out)


# ------------------------------------------------------------------ 5. epsilon (gradient safety)
def _dup_activations(T=64, H=4, D=16, B_=8, seed=0):
    """Real-ish q/k at LM initialisation with exact duplicates injected (S = 0 pairs)."""
    cfg = train.make_cfg("lm", "lp_3", qk_dim=D)
    m = train.build_model(cfg, seed=seed)
    data = train.get_lm_data(cfg["T"])
    x, _ = data.batch(np.random.default_rng(1), B_)
    with torch.no_grad():
        h = m.tok_emb(x) + m.pos_emb(torch.arange(T))
        blk = m.blocks[0]
        q = blk.attn.W_q(blk.ln1(h)).view(B_, T, H, D).transpose(1, 2).contiguous()
    k = q.clone()
    return q, k


def stage_epsilon(out=None, quick=False, **_):
    """Gradient safety of the relative epsilon r: no NaN/inf, float32 accuracy against float64, and the distortion of the
    metric. Exact duplicates of q are injected as keys, which gives sums of zero. (Training sensitivity to r was measured on the
    recall task and is part of the shelf: mqar, stage epsilon_runs.)"""
    hdr("epsilon: gradient safety of r")
    res = {"r_grid": [0.0, 1e-12, 1e-9, 1e-6, 1e-4, 1e-2], "fp32_vs_fp64": [], "distortion": [], "Ds": list(LM_DS[:1] if quick else LM_DS)}
    for D in res["Ds"]:
        q, k = _dup_activations(D=D)
        q64, k64 = q.double(), k.double()
        g = torch.randn(q.shape[0], q.shape[1], q.shape[2], q.shape[2])
        for p in (1.0, 1.5, 2.0, 3.0, 4.0, 8.0):
            S_typ = float(lp.power_sum(q64, k64, p).mean())
            for r in res["r_grid"]:
                eps = r * S_typ

                def run(dt, qq, kk, gg, e):
                    a = qq.clone().requires_grad_(True); b = kk.clone().requires_grad_(True)
                    d = lp.lp_dist(a, b, p, e, "auto" if dt == torch.float32 else "naive", 16, False)
                    (d * gg).sum().backward()
                    return d.detach(), a.grad, b.grad
                d32 = run(torch.float32, q, k, g, eps)
                nonfinite = int(sum((~torch.isfinite(t)).sum() for t in d32))
                # the duplicate pair has a distance set entirely by eps, so compare against the same eps in float64
                ref = run(torch.float64, q64, k64, g.double(), eps if r > 0 else 1e-300)
                rel = [float((x.double() - y).abs().max() / (y.abs().max() + 1e-300)) for x, y in zip(d32, ref)]
                res["fp32_vs_fp64"].append({"D": D, "p": p, "r": r, "nonfinite": nonfinite, "rel_err_d_dq_dk": rel})
            for r in res["r_grid"][1:]:
                res["distortion"].append({"D": D, "p": p, "r": r, "floor_ratio": r ** (1 / p), "typical_relative_shift": r / p})
    print("fp32 failures (non-finite) and worst relative error by (D, p, r):")
    for row in res["fp32_vs_fp64"]:
        print(f"  D={row['D']:<3} p={row['p']:<4} r={row['r']:<7g} nonfinite={row['nonfinite']} worst rel err={max(row['rel_err_d_dq_dk']):.2e}")
    return _save("epsilon", res, out)


# ------------------------------------------------------------------ helpers for run stages
# ------------------------------------------------------------------ helpers for run stages
def _rows(run_dir):
    p = Path(run_dir) / "log.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []



def _summarise_trial(run_dir, task="lm"):
    rows = _rows(run_dir)
    tr = [r for r in rows if "train_loss" in r]
    dg = [r for r in rows if "layers" in r]
    key = "val_bpc" if task == "lm" else "val_acc"
    ev_ = [r for r in rows if key in r]
    s = {"final_train_loss": tr[-1]["train_loss"] if tr else None, "first_train_loss": tr[0]["train_loss"] if tr else None,
         "final_" + key: ev_[-1][key] if ev_ else None, "grad_norm_max": max((r["grad_norm_max"] for r in tr), default=None),
         "grad_norm_mean": float(np.mean([r["grad_norm_mean"] for r in tr])) if tr else None,
         "frac_log_points_clipped": float(np.mean([r["grad_norm_max"] > 1.0 for r in tr])) if tr else None,
         "max_loss_jump": max((tr[i]["train_loss"] - tr[i - 1]["train_loss"] for i in range(1, len(tr))), default=0.0)}
    if dg:
        first, last = dg[0]["layers"], dg[-1]["layers"]
        s["max_abs_logit_max_over_time"] = max(max(l["max_abs_logit"]) for r in dg for l in r["layers"])
        s["max_abs_logit_final"] = max(max(l["max_abs_logit"]) for l in last)
        s["entropy_final_mean"] = float(np.mean([np.mean(l["entropy"]) for l in last]))
        s["entropy_uniform"] = float(np.mean([l["entropy_uniform"] for l in last]))
        s["q_norm_growth"] = float(np.mean([np.mean(l1["q_norm_mean"]) / max(np.mean(l0["q_norm_mean"]), 1e-12) for l0, l1 in zip(first, last)]))
        s["beta_final"] = [l["beta"] for l in last]
        s["beta_first"] = [l["beta"] for l in first]
        s["max_abs_logit_trace"] = [max(max(l["max_abs_logit"]) for l in r["layers"]) for r in dg]
    return s


# ------------------------------------------------------------------ 6. learning-rate safeguard
def stage_lr(out=None, quick=False, workers=1, threads=1, base_lr=None, mult=3.0, lm_steps=400, **_):
    """Each arm at the intended rate and at mult x that rate (truncated runs on the full-length schedule)."""
    lr_lm = base_lr or train.make_cfg("lm", "dot_tied")["lr"]
    hdr(f"learning-rate safeguard: LM {lr_lm} and {lr_lm * mult:g}")
    root = Path(out or PILOT) / "lr_safeguard"
    arms = ["dot_tied", "lp_2", "lp_inf"] if quick else ARMS
    cfgs = []
    for k in (1, mult):
        for a in arms:
            lr = lr_lm * k
            cfgs.append(train.make_cfg("lm", a, lr=lr, tag=f"lr{lr:g}", steps=2500, stop_after=lm_steps if not quick else 50,
                                       eval_every=50 if quick else 100, diag_every=50 if quick else 100, ckpt_every=10_000,
                                       threads=threads, eval_windows_mid=200))
    res = runner.run_many(cfgs, root, workers=workers)
    table = []
    for cfg, r in zip(cfgs, res):
        if "error" in r:
            print("ERROR", r["run_id"], r["error"][-300:]); continue
        s = _summarise_trial(root / train.run_id(cfg), "lm")
        s.update(task="lm", arm=cfg["arm"], lr=cfg["lr"], diverged=r["diverged"], diverged_step=r["diverged_step"])
        table.append(s)
    out_ = {"lm_lrs": [lr_lm, lr_lm * mult], "lm_steps": lm_steps, "table": table}
    for s in table:
        print(f"  {s['arm']:10s} lr={s['lr']:.0e} div={s['diverged']} train_loss={s['final_train_loss']:.3f} final_val_bpc={s['final_val_bpc']:.3f} "
              f"max|logit|={s.get('max_abs_logit_max_over_time', float('nan')):.1f} gradnorm_max={s['grad_norm_max']:.2f} "
              f"clipped={s['frac_log_points_clipped']:.2f} q_norm_growth={s.get('q_norm_growth', float('nan')):.2f}")
    return _save("lr_safeguard", out_, out)


# ------------------------------------------------------------------ 7. seed variance (paired differences)
def mde_in_sd_units(n, alpha=0.05, power=0.8):
    """Minimal detectable paired mean difference in units of the sd of the paired difference (two-sided t test)."""
    from scipy import stats
    df = n - 1
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    lo, hi = 0.0, 50.0
    for _ in range(80):
        mid = (lo + hi) / 2
        pw = 1 - stats.nct.cdf(tcrit, df, mid * math.sqrt(n)) + stats.nct.cdf(-tcrit, df, mid * math.sqrt(n))
        lo, hi = (mid, hi) if pw < power else (lo, mid)
    return (lo + hi) / 2


def paired_summary(vals, control, arm):
    """vals[arm] = list of per-seed metrics (same seed order). Returns paired-difference statistics."""
    from scipy import stats
    a, c = np.array(vals[arm], float), np.array(vals[control], float)
    d = a - c
    n = len(d)
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    tq = stats.t.ppf(0.975, n - 1)
    return {"arm": arm, "n": n, "mean_diff": float(d.mean()), "sd_diff": sd, "ci95": [float(d.mean() - tq * se), float(d.mean() + tq * se)],
            "diffs": d.tolist(), "sd_arm": float(a.std(ddof=1)), "sd_control": float(c.std(ddof=1)),
            "corr_across_seeds": float(np.corrcoef(a, c)[0, 1]) if n > 2 and a.std() > 0 and c.std() > 0 else None,
            "mde_abs_uncorrected": mde_in_sd_units(n, 0.05) * sd, "mde_abs_holm7": mde_in_sd_units(n, 0.05 / 7) * sd}


def stage_seedvar(out=None, quick=False, workers=1, threads=1, n_seeds=5, lm_arms=None, lm_steps=None, **_):
    """Seeds are paired across arms (shared initialisation of common parameters and shared data stream), full-length configuration."""
    hdr("seed variance (paired differences)")
    root = Path(out or PILOT) / "seedvar"
    lm_arms = lm_arms or (["dot_tied", "lp_2"] if quick else ["dot_tied", "lp_2", "lp_inf"])
    seeds = list(range(100, 100 + (2 if quick else n_seeds)))
    lm_kw = dict(diag_every=500, eval_every=250)
    if lm_steps:
        lm_kw["steps"] = lm_steps
    if quick:
        lm_kw.update(steps=30, eval_every=30, diag_every=30, eval_windows_mid=32)
    cfgs = [train.make_cfg("lm", a, seed=s_, threads=threads, **lm_kw) for s_ in seeds for a in lm_arms]
    res = runner.run_many(cfgs, root, workers=workers)
    vals, walls, diverged = {}, {}, {}
    for cfg, r in zip(cfgs, res):
        if "error" in r:
            print("ERROR", r["run_id"], r["error"][-300:]); continue
        vals.setdefault(cfg["arm"], []).append(r["final_val"]["val_bpc"])
        walls.setdefault(cfg["arm"], []).append(r["wall_seconds"])
        diverged.setdefault(cfg["arm"], []).append(r["diverged"])
    summ = {"metric": "val_bpc", "values": vals, "wall_seconds_mean": {a: float(np.mean(w)) for a, w in walls.items()},
            "diverged": diverged, "single_arm_sd": {a: float(np.std(v, ddof=1)) for a, v in vals.items() if len(v) > 1}, "paired": {}}
    for a in lm_arms:
        if a != "dot_tied" and a in vals and "dot_tied" in vals and len(vals[a]) == len(vals["dot_tied"]) > 1:
            summ["paired"][a] = paired_summary(vals, "dot_tied", a)
    print("--- lm: validation bpc ---")
    for a in lm_arms:
        if a in vals:
            print(f"  {a:10s} mean={np.mean(vals[a]):.4f} sd={np.std(vals[a], ddof=1):.4f} wall={np.mean(walls[a]):.0f}s")
    for a, p in summ["paired"].items():
        print(f"  paired {a} - dot_tied: mean={p['mean_diff']:+.4f} sd_diff={p['sd_diff']:.4f} corr={p['corr_across_seeds']}")
    return _save("seedvar", {"seeds": seeds, "tasks": {"lm": summ}}, out)


# ------------------------------------------------------------------ 8. budget
def stage_budget(out=None, total_hours=48.0, laptop_factor=1.0, Ds=(2, 4, 16), seeds=(3, 4, 5), **_):
    """Run-time projection for the language-model plan: nine arms plus the uniform control at each head size.
    Per-run seconds come from the seedvar stage (measured arms) and the bench stage (ratios for the other arms);
    laptop_factor scales this machine's times to the laptop (re-measure by running bench and seedvar there)."""
    hdr("budget projection")
    sv, bn = _load("seedvar", out), _load("bench", out)
    if not sv:
        print("seedvar results missing"); return None
    lm = sv["tasks"]["lm"]["wall_seconds_mean"]
    lm_est = dict(lm)
    if bn:
        step = {r["arm"]: r["sec_per_step"] for r in bn["lm_arms"] if "arm" in r and "sec_per_step" in r}
        ref = [a for a in lm if a in step]
        if ref:
            ratio = np.mean([lm[a] / step[a] for a in ref])      # bench steps used 2 threads, seedvar runs 1 thread
            for a in ARMS:
                if a not in lm_est and a in step:
                    lm_est[a] = step[a] * ratio
    res = {"laptop_factor": laptop_factor, "total_hours": total_hours, "lm_seconds_per_run": lm_est,
           "note": "seconds per run in a single-thread process while another run shares the machine; two workers assumed"}
    scen = {}
    if all(a in lm_est for a in ARMS):
        per_seed = sum(lm_est[a] for a in ARMS)
        for n in seeds:
            hours = per_seed * len(Ds) * n * laptop_factor / 3600 / 2.0
            scen[f"{len(Ds)}_head_sizes_x_{n}_seeds"] = {"hours_two_workers": hours, "fits": hours <= total_hours}
    res["scenarios"] = scen
    print("LM seconds/run:", {a: round(v) for a, v in lm_est.items()})
    for k, v in scen.items():
        print(f"  {k}: {v['hours_two_workers']:.1f} h (two workers; the uniform control and any learning-rate scans are extra)")
    return _save("budget", res, out)


# ------------------------------------------------------------------ 9. learning-rate scan (full length, head size 16)
def stage_lr_scan(out=None, quick=False, workers=1, threads=1, lrs=(1e-3, 3e-3, 1e-2), **_):
    """Full-length runs at several rates for three arms: is the 'conservative' rate under-training some arm?
    (A shared rate that starves one arm would turn an optimisation artefact into an apparent capacity difference.)"""
    hdr("learning-rate scan (full length)")
    root = Path(out or PILOT) / "lr_scan"
    kw = dict(diag_every=500, eval_every=250)
    if quick:
        kw = dict(steps=30, eval_every=30, diag_every=30, eval_windows_mid=32)
    cfgs = [train.make_cfg("lm", a, lr=lr, tag=f"lr{lr:g}", threads=threads, **kw)
            for lr in lrs for a in (["dot_tied"] if quick else ["dot_tied", "lp_2", "lp_inf"])]
    res = runner.run_many(cfgs, root, workers=workers)
    rows = []
    for cfg, r in zip(cfgs, res):
        if "error" in r:
            print("ERROR", r["run_id"], r["error"][-300:]); continue
        row = {"arm": cfg["arm"], "lr": cfg["lr"], "steps": cfg["steps"], "diverged": r["diverged"], "wall": r["wall_seconds"],
               "val_bpc": r["final_val"]["val_bpc"]}
        rows.append(row)
        print("  ", json.dumps(row))
    return _save("lr_scan", {"rows": rows}, out)



def _lm_lr():
    """LM rate set with --lr-lm; otherwise 3e-3 (the rate proposed in the pilot report; not yet a user decision)."""
    return train.DEFAULTS["lm"].get("lr", 3e-3)


# ------------------------------------------------------------------ 11. language model with a small query-key dimension
# Not part of the original pilot plan. Added at the user's request (2026-10-02): the LM at D = 16 shows no separation between
# arms, so we shrink only the query-key width per head (values stay 16 per head) and add the uniform-attention control.
def stage_lm_small_d_lr(out=None, quick=False, workers=1, threads=1, Ds=None, lrs=(3e-3, 1e-2, 3e-2), n_null_seeds=5, **_):
    """Learning-rate scan at each small D (tied dot and L2, one seed, full length) plus the uniform-attention control
    (a rate scan, and 5 paired seeds at the current LM rate for the dynamic range at D = 16)."""
    hdr("LM small query-key dimension: learning-rate scan and null arm")
    root = Path(out or PILOT) / "lm_small_d_lr"
    Ds = Ds or [2, 4, 8, 12, 16]
    kw = dict(threads=threads, diag_every=500, eval_every=250)
    if quick:
        kw.update(steps=30, eval_every=30, diag_every=30, eval_windows_mid=32)
        Ds, lrs, n_null_seeds = Ds[:1], lrs[:1], 1
    cfgs = []
    for D in Ds:
        for lr in lrs:
            for a in ("dot_tied", "lp_2"):
                cfgs.append(train.make_cfg("lm", a, lr=lr, qk_dim=D, tag=f"lr{lr:g}", **kw))
    for lr in lrs:                    # the null arm does not depend on D; one run per rate
        cfgs.append(train.make_cfg("lm", NULL_ARM, lr=lr, tag=f"lr{lr:g}", **kw))
    for s_ in range(100, 100 + n_null_seeds):      # same seeds as the seedvar stage, at the current LM rate
        cfgs.append(train.make_cfg("lm", NULL_ARM, seed=s_, lr=_lm_lr(), tag=f"lr{_lm_lr():g}", **kw))
    res = runner.run_many(cfgs, root, workers=workers)
    rows = []
    for cfg, r in zip(cfgs, res):
        if "error" in r:
            print("ERROR", r["run_id"], r["error"][-300:]); continue
        rows.append({"arm": cfg["arm"], "qk_dim": cfg.get("qk_dim"), "seed": cfg["seed"], "lr": cfg["lr"], "diverged": r["diverged"],
                     "wall": r["wall_seconds"], "val_bpc": r["final_val"]["val_bpc"] if not r["diverged"] else None})
    print("validation bpc (seed 0 unless noted):")
    for row in rows:
        print("  ", json.dumps(row))
    return _save("lm_small_d_lr", {"rows": rows}, out)


def stage_lm_small_d(out=None, quick=False, workers=1, threads=1, Ds=None, n_seeds=5, arms=None, **_):
    """Main round: all nine arms plus the uniform control, small query-key dimension, paired seeds, one shared LM rate
    (the rate in train.DEFAULTS['lm']['lr'], set with --lr-lm). The null arm does not depend on D and runs once per seed."""
    lr = _lm_lr()
    hdr(f"LM small query-key dimension: main round (lr {lr:g})")
    root = Path(out or PILOT) / "lm_small_d"
    Ds = Ds or [2, 4, 8, 12, 16]
    arms = arms or list(ARMS)
    seeds = list(range(100, 100 + n_seeds))
    kw = dict(threads=threads, diag_every=500, eval_every=250, lr=lr, tag=f"lr{lr:g}")
    if quick:
        kw.update(steps=30, eval_every=30, diag_every=30, eval_windows_mid=32)
        Ds, arms, seeds = Ds[:1], ["dot_tied", "lp_2"], seeds[:2]
    cfgs = []
    for D in Ds:
        for s_ in seeds:
            for a in arms:
                cfgs.append(train.make_cfg("lm", a, seed=s_, qk_dim=D, **kw))
    for s_ in seeds:
        cfgs.append(train.make_cfg("lm", NULL_ARM, seed=s_, **kw))
    res = runner.run_many(cfgs, root, workers=workers)
    vals = {}      # (qk_dim or "null") -> arm -> per-seed bpc, in seed order
    walls = {}
    for cfg, r in zip(cfgs, res):
        if "error" in r:
            print("ERROR", r["run_id"], r["error"][-300:]); continue
        if r["diverged"]:
            print("DIVERGED", r["run_id"]); continue
        key = "null" if cfg["arm"] == NULL_ARM else cfg["qk_dim"]
        vals.setdefault(key, {}).setdefault(cfg["arm"], []).append(r["final_val"]["val_bpc"])
        walls.setdefault(cfg["arm"], []).append(r["wall_seconds"])
    summary = {"lr": lr, "seeds": seeds, "per_D": {}}
    null = vals.get("null", {}).get(NULL_ARM)
    for D in Ds:
        v = vals.get(D, {})
        S = {"values": v, "mean": {a: float(np.mean(x)) for a, x in v.items()}, "sd": {a: float(np.std(x, ddof=1)) for a, x in v.items() if len(x) > 1},
             "paired_vs_dot_tied": {}, "paired_vs_uniform": {}}
        for a in v:
            if a != "dot_tied" and "dot_tied" in v and len(v[a]) == len(v["dot_tied"]) > 1:
                S["paired_vs_dot_tied"][a] = paired_summary(v, "dot_tied", a)
            if null and len(v[a]) == len(null) > 1:
                S["paired_vs_uniform"][a] = paired_summary({**v, NULL_ARM: null}, NULL_ARM, a)
        summary["per_D"][str(D)] = S
        print(f"--- qk_dim D = {D}: validation bpc, mean (sd) over {len(seeds)} seeds ---")
        for a in arms:
            if a in v:
                p = S["paired_vs_dot_tied"].get(a)
                tail = f"  vs dot_tied {p['mean_diff']:+.4f} (sd {p['sd_diff']:.4f})" if p else ""
                print(f"  {a:10s} {S['mean'][a]:.4f} ({S['sd'].get(a, float('nan')):.4f}){tail}")
    if null:
        summary["uniform"] = {"values": null, "mean": float(np.mean(null)), "sd": float(np.std(null, ddof=1)) if len(null) > 1 else None}
        print(f"  uniform attention (null, any D): {np.mean(null):.4f}")
    return _save("lm_small_d", summary, out)
