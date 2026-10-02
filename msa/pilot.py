"""Pilot stages. Each stage writes results/pilot/<stage>.json (and figures) and returns its result dict.
Called from execute_project.py; every stage can also be re-run on its own."""
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import bench as B
from . import gradcheck as GC
from . import lp, mqar_gen, runner, train
from . import calibration as cal
from .arms import ARMS, LP_ARMS

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT / "results" / "pilot"


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
    # smoke: one training step per task and arm family
    t0 = time.time()
    for task in ("lm", "mqar"):
        for arm in ("dot_tied", "lp_3", "lp_inf"):
            cfg = train.make_cfg(task, arm, steps=2, eval_every=2, diag_every=2, ckpt_every=100, log_every=1, threads=1,
                                 **({"eval_windows_mid": 8} if task == "lm" else {"n_val": 200, "n_test": 200, "n_val_mid": 100}))
            import shutil, tempfile
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


# ------------------------------------------------------------------ 3. generator check
def stage_generator(out=None, **_):
    hdr("MQAR generator check")
    res = {"cells": [], "ok": True}
    for P in (8, 16, 32, 64):
        r = mqar_gen.check_generator(P=P, Q=P // 2, n_keys=256, n_vals=64, n=20000, seed=P)
        res["cells"].append(r)
        res["ok"] &= r["ok"]
        v = r["vectorised"]
        print(f"P={P:2d} Q={P//2:2d}: ok={r['ok']} chance sim/theory={v['chance_sim']:.4f}/{v['chance_theory']:.4f} "
              f"key-blind sim/theory={v['key_blind_sim']:.4f}/{v['key_blind_theory']:.4f}")
    # equivalence of the fixed evaluation sets across arms: same seed -> identical arrays
    a = mqar_gen.fixed_set(777001 + 16, 100, 16, 8, 256, 64)
    b = mqar_gen.fixed_set(777001 + 16, 100, 16, 8, 256, 64)
    res["fixed_sets_identical"] = all((x == y).all() for x, y in zip(a, b))
    # model-level sanity on untrained models (the trained-model versions are recorded in every run)
    from .models import MQARModel, init_model
    from . import evaluate as ev
    ctx, qry, tgt = mqar_gen.fixed_set(1, 2000, 16, 8, 256, 64)
    sanity = {}
    for arm in ("dot_tied", "lp_2"):
        m = init_model(MQARModel(arm), 0)
        sanity[arm] = ev.mqar_sanity(m, ctx, qry, tgt)
    res["untrained_model_sanity"] = sanity
    res["shuffle_invariant"] = all(s["shuffle_abs_diff"] < 1e-9 for s in sanity.values())
    res["no_context_at_chance"] = all(abs(s["acc_no_context"] - s["chance"]) < 0.01 for s in sanity.values())
    res["ok"] &= res["fixed_sets_identical"] and res["shuffle_invariant"] and res["no_context_at_chance"]
    print(f"fixed sets identical: {res['fixed_sets_identical']}; shuffle invariant: {res['shuffle_invariant']}; "
          f"no-context at chance: {res['no_context_at_chance']}")
    return _save("generator", res, out)


# ------------------------------------------------------------------ 4. timing and memory
def _fmt_bench(r):
    if "error" in r:
        return "ERR"
    return f"{r['sec_per_step']:.3f}s/{r['peak_rss_mb']:.0f}MB"


def stage_bench(out=None, quick=False, threads=2, time_budget=None, fresh=False, **_):
    """Timing/memory per arm at the real model sizes for both tasks, plus implementation/chunk comparisons.
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
    res = {"threads": threads, "kernel_ref": [], "chunk_sweep": [], "lm_arms": [], "lm_impls": [], "mqar_arms": [],
           "threads_cmp": [], "parallel": {}}
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
    for task, extra in (("lm", []), ("mqar", ["--P", 64, "--D", 16])):
        for arm in (["lp_3"] if quick else ["lp_1.5", "lp_3", "lp_inf"]):
            for chunk in ([4, 16] if quick else [2, 4, 8, 16, 32]):
                r = B.measure("model", "--task", task, "--arm", arm, "--chunk", chunk, "--B", 64 if task == "lm" else 128,
                              "--threads", 1, "--reps", 4 if quick else 12, *extra)
                r.update(task=task, arm=arm, chunk=chunk)
                res.setdefault("chunk_sweep_model", []).append(r)
                print(f"  model chunk sweep {task} {arm:7s} chunk={chunk:<3}: {_fmt_bench(r)}", flush=True)
    # (c) full training step per arm, language model (B=64, T=64, 2 layers, 4 heads of 16)
    for arm in (["dot_tied", "lp_3"] if quick else ARMS):
        r = B.measure("model", "--task", "lm", "--arm", arm, "--B", 64, "--threads", threads, "--reps", reps)
        r.update(arm=arm)
        res["lm_arms"].append(r)
        print(f"  LM step arm={arm:10s}: {_fmt_bench(r)}  (params {r.get('n_params')})", flush=True)
    for impl in ([] if quick else ["naive", "ckpt", "custom"]):
        for arm in ("lp_3", "lp_inf"):
            r = B.measure("model", "--task", "lm", "--arm", arm, "--impl", impl, "--B", 64, "--threads", threads, "--reps", reps)
            r.update(arm=arm, impl=impl)
            res["lm_impls"].append(r)
            print(f"  LM step arm={arm:7s} impl={impl:6s}: {_fmt_bench(r)}", flush=True)
    # (d) recall task per arm at several (P, D) (B=128, Q=P/2)
    for P, D in ([(64, 16)] if quick else [(64, 16), (32, 8), (16, 4)]):
        for arm in (["dot_tied", "lp_3"] if quick else ARMS):
            r = B.measure("model", "--task", "mqar", "--arm", arm, "--P", P, "--D", D, "--B", 128, "--threads", threads, "--reps", reps)
            r.update(arm=arm, P=P, D=D)
            res["mqar_arms"].append(r)
            print(f"  MQAR step P={P} D={D} arm={arm:10s}: {_fmt_bench(r)}", flush=True)
    # (e) threads and process-level parallelism (LM, lp_3 and dot_tied)
    if not quick and (os.cpu_count() or 1) >= 2:
        import subprocess
        for arm in ("lp_3", "dot_tied"):
            for th in sorted({1, threads}):
                r = B.measure("model", "--task", "lm", "--arm", arm, "--B", 64, "--threads", th, "--reps", reps)
                r.update(arm=arm, threads=th)
                res["threads_cmp"].append(r)
                print(f"  LM {arm} threads={th}: {_fmt_bench(r)}", flush=True)
        # two concurrent single-thread processes vs one process with all threads
        n = min(threads, os.cpu_count() or 1)
        cmd = [sys.executable, "-m", "msa.bench", "model", "--task", "lm", "--arm", "lp_3", "--B", "64", "--threads", "1", "--reps", str(reps)]
        procs = [subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, cwd=ROOT) for _ in range(n)]
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


# ------------------------------------------------------------------ 5. calibration
def stage_calibrate(out=None, target=1.0, r=None, **_):
    """Calibration constants c_p, typical sums inside the root, and the epsilon implied by r (provisional r)."""
    hdr("calibration at initialisation")
    r = train.DEFAULTS["common"]["r"] if r is None else r
    res = {"target": target, "r": r, "lm": {}, "mqar": {}}
    for arm in ARMS:
        cfg = train.make_cfg("lm", arm, target=target, r=r)
        res["lm"][arm] = train.get_constants(cfg)
    for D in (2, 4, 8, 16):
        for P in ((16, 64) if D == 16 else (16,)):
            cell = {}
            for arm in ARMS:
                cfg = train.make_cfg("mqar", arm, P=P, D=D, target=target, r=r)
                cell[arm] = train.get_constants(cfg)
            res["mqar"][f"P{P}_D{D}"] = cell
    # sanity: the tied dot product constant should be near 1/sqrt(D)
    res["dot_sanity"] = {"lm": {"c_layers": [l["c"] for l in res["lm"]["dot_tied"]], "one_over_sqrt_D": 1 / math.sqrt(16)},
                         "mqar": {k: {"c": v["dot_tied"][0]["c"], "one_over_sqrt_D": 1 / math.sqrt(int(k.split("_D")[1]))}
                                  for k, v in res["mqar"].items()}}
    print(f"target logit spread = {target}, provisional r = {r}")
    print("LM (per layer): arm, c, S_typ (mean sum inside the root), S_median, eps")
    for arm in ARMS:
        for l in res["lm"][arm]:
            print(f"  {arm:10s} L{l['layer']}: c={l['c']:.4f} std_raw={l['std_raw']:.4f}"
                  + (f" S_typ={l['S_typ']:.4g} S_med={l['S_median']:.4g} eps={l['eps']:.3g}" if "S_typ" in l else ""))
    print("dot_tied sanity c vs 1/sqrt(D):", json.dumps(res["dot_sanity"]))
    return _save("calibrate", res, out)


# ------------------------------------------------------------------ 6. epsilon (choice of r)
def _dup_activations(T=64, H=4, D=16, B_=8, seed=0):
    """Real-ish q/k at LM initialisation with exact duplicates injected (S = 0 pairs)."""
    cfg = train.make_cfg("lm", "lp_3")
    m = train.build_model(cfg, seed=seed)
    data = train.get_lm_data(cfg["T"])
    x, _ = data.batch(np.random.default_rng(1), B_)
    with torch.no_grad():
        h = m.tok_emb(x) + m.pos_emb(torch.arange(T))
        blk = m.blocks[0]
        q = blk.attn.W_q(blk.ln1(h)).view(B_, T, H, D).transpose(1, 2).contiguous()
    k = q.clone()
    return q, k


def stage_epsilon(out=None, quick=False, workers=1, threads=1, **_):
    """Choose r: gradient safety (no NaN/inf, fp32 accuracy), distortion of the metric, and the scale of
    the sums inside the root that trained models actually produce for matched pairs."""
    hdr("epsilon: choice of r")
    res = {"r_grid": [0.0, 1e-12, 1e-9, 1e-6, 1e-4, 1e-2], "fp32_vs_fp64": [], "distortion": [], "trained": []}
    q, k = _dup_activations()
    q64, k64 = q.double(), k.double()
    # k contains exact duplicates of q at the same positions: S = 0 on those pairs
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
            ref = run(torch.float64, q64, k64, g.double(), eps if r > 0 else 1e-300)   # near-exact reference
            d32 = run(torch.float32, q, k, g, eps)
            nonfinite = int(sum((~torch.isfinite(t)).sum() for t in d32))
            # reference where the diagonal (exact duplicate) is excluded from the error: the duplicate pair has a
            # distance set entirely by eps, so compare against the same eps in float64
            ref_e = run(torch.float64, q64, k64, g.double(), eps) if r > 0 else ref
            rel = [float((x.double() - y).abs().max() / (y.abs().max() + 1e-300)) for x, y in zip(d32, ref_e)]
            res["fp32_vs_fp64"].append({"p": p, "r": r, "nonfinite": nonfinite, "rel_err_d_dq_dk": rel})
        for r in res["r_grid"][1:]:
            res["distortion"].append({"p": p, "r": r, "floor_ratio": r ** (1 / p),   # dist at an exact match / typical-scale dist
                                      "typical_relative_shift": r / p})
    print("fp32 failures (non-finite) and worst relative error by (p, r):")
    for row in res["fp32_vs_fp64"]:
        print(f"  p={row['p']:<4} r={row['r']:<7g} nonfinite={row['nonfinite']} worst rel err={max(row['rel_err_d_dq_dk']):.2e}")
    # trained-model scale of matched pairs (needs MQAR final weights from the transition stage)
    root = Path(out or PILOT) / "lr_safeguard"      # short trained MQAR models (base rate) at the pilot cell, all L^p arms
    if root.exists():
        for f in sorted(root.glob("mqar_*_lp_*_lr0.001/final.pt")):
            ck = torch.load(f, weights_only=False)
            cfg = ck["cfg"]
            if cfg["arm"] in ("lp_inf",):
                continue
            m = train.build_model(cfg)
            m.load_state_dict(ck["state_dict"])
            p = m.scorer.p
            val, _ = train._mqar_sets(cfg)
            c_, q_, t_ = (torch.from_numpy(x[:500]) for x in val)
            with torch.no_grad():
                _, att, logits, zk = m(c_, q_, return_att=True)
                xq = m.key_emb(q_) + m.val_emb.weight[m.n_vals]
                zq = m.W_q(m.ln_in(xq))
                S = lp.power_sum(zq.unsqueeze(1), zk.unsqueeze(1), p)[:, 0]          # B,Q,P
                match = (c_[:, None, :, 0] == q_[:, :, None])
                S_match = S[match]; S_rest = S[~match]
            S_typ0 = train.get_constants(cfg)[0]["S_typ"]
            res["trained"].append({"arm": cfg["arm"], "S_typ_init": S_typ0, "S_match_median": float(S_match.median()),
                                   "S_match_q01": float(torch.quantile(S_match, 0.01)), "S_match_min": float(S_match.min()),
                                   "S_other_median": float(S_rest.median()),
                                   "match_over_Styp_init_q01": float(torch.quantile(S_match, 0.01)) / S_typ0})
            print(f"  trained {cfg['arm']:7s}: S_typ(init)={S_typ0:.3g} matched-pair S median={float(S_match.median()):.3g} "
                  f"1st pct={float(torch.quantile(S_match, 0.01)):.3g} min={float(S_match.min()):.3g}")
    # training sensitivity: does r (and the p = 2 implementation path) change the outcome at the pilot cell?
    cell = pilot_cell(out)
    root = Path(out or PILOT) / "epsilon_runs"
    cfgs = []
    seeds = (0,) if quick else (0, 1)
    for arm in (["lp_8"] if quick else ["lp_2", "lp_4", "lp_8"]):
        for r_ in (1e-12, 1e-6, 1e-3):
            for sd in seeds:
                cfgs.append(train.make_cfg("mqar", arm, r=r_, tag=f"r{r_:g}", seed=sd, steps=3000 if not quick else 100, threads=threads, diag_every=10_000, **_mqar_kw(cell)))
    for sd in seeds:        # p = 2: matmul expansion vs direct differences (custom kernel), same r
        cfgs.append(train.make_cfg("mqar", "lp_2", r=1e-12, impl="custom", tag="r1e-12_direct", seed=sd, steps=3000 if not quick else 100, threads=threads, diag_every=10_000, **_mqar_kw(cell)))
    results = runner.run_many(cfgs, root, workers=workers)
    res["training_sensitivity"] = []
    for cfg, r_ in zip(cfgs, results):
        if "error" in r_:
            print("ERROR", r_["run_id"], r_["error"][-300:]); continue
        row = {"arm": cfg["arm"], "r": cfg["r"], "impl": cfg["impl"], "seed": cfg["seed"], "test_acc": r_["final_test"]["acc"],
               "steps_to_90": r_["steps_to_90"], "wall": r_["wall_seconds"]}
        res["training_sensitivity"].append(row)
        print("  ", json.dumps(row))
    res["mqar_cell"] = cell
    return _save("epsilon", res, out)


# ------------------------------------------------------------------ helpers for run stages
def _rows(run_dir):
    p = Path(run_dir) / "log.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def _mqar_kw(cell):
    return {"P": cell["P"], "D": cell["D"]}


def _transition_name():
    """Transition results are kept per learning rate: 'transition' for the original 1e-3, 'transition_lr<lr>' otherwise."""
    lr = train.make_cfg("mqar", "dot_tied")["lr"]
    return "transition" if lr == 1e-3 else f"transition_lr{lr:g}"


def pilot_cell(out=None):
    """The (P, D) cell used for the MQAR safeguard and seed-variance runs: the one proposed by the transition stage."""
    t = _load(_transition_name(), out)
    if t and t.get("pilot_cell"):
        return t["pilot_cell"]
    return {"P": 32, "D": 4}


def transition_summary(cells, arms, Ps, Ds, thr=0.9):
    """Per arm: for each D the pair count at which accuracy falls below `thr` (log-linear interpolation in P;
    '<Pmin' / '>Pmax' when the crossing lies outside the grid), and for each P the smallest D with accuracy >= thr."""
    out = {"threshold": thr, "P_cross": {}, "D_min_at_P": {}}
    for a in arms:
        out["P_cross"][a] = {}
        for D in Ds:
            acc = [(P, cells[f"P{P}_D{D}"]["arms"][a]["test_acc"]) for P in Ps if f"P{P}_D{D}" in cells and a in cells[f"P{P}_D{D}"]["arms"]]
            if not acc:
                continue
            if acc[0][1] < thr:
                out["P_cross"][a][D] = f"<{acc[0][0]}"
            elif acc[-1][1] >= thr:
                out["P_cross"][a][D] = f">{acc[-1][0]}"
            else:
                for (P0, a0), (P1, a1) in zip(acc[:-1], acc[1:]):
                    if a0 >= thr > a1:
                        f = (a0 - thr) / (a0 - a1)
                        out["P_cross"][a][D] = float(math.exp(math.log(P0) + f * (math.log(P1) - math.log(P0))))
                        break
        out["D_min_at_P"][a] = {}
        for P in Ps:
            ok = [D for D in Ds if f"P{P}_D{D}" in cells and a in cells[f"P{P}_D{D}"]["arms"] and cells[f"P{P}_D{D}"]["arms"][a]["test_acc"] >= thr]
            out["D_min_at_P"][a][P] = min(ok) if ok else f">{max(Ds)}"
    return out


# ------------------------------------------------------------------ 7. recall transition
def stage_transition(out=None, quick=False, workers=1, threads=1, steps=3000, arms=None, Ps=None, Ds=None, seed=0, name=None, n_keys=256, **_):
    """One seed per (arm, P, D) cell: where does accuracy start to drop, for each arm?"""
    name = name or _transition_name()
    hdr(f"recall transition scan ({name})")
    arms = arms or (["dot_tied", "lp_2"] if quick else ["dot_tied", "lp_1", "lp_2", "lp_inf", "dot_untied"])
    Ps = Ps or ([16, 32] if quick else [8, 16, 32, 64])
    Ds = Ds or ([2, 4] if quick else [2, 3, 4, 6, 8])
    lr_ = train.make_cfg("mqar", "dot_tied")["lr"]
    root = Path(out or PILOT) / name
    cfgs = [train.make_cfg("mqar", a, P=P, D=D, steps=steps, seed=seed, threads=threads, diag_every=steps, n_keys=n_keys)
            for D in Ds for P in Ps for a in arms]
    res = runner.run_many(cfgs, root, workers=workers)
    for r in res:
        if "error" in r:
            print("ERROR", r["run_id"], r["error"][-300:])
    cells = {}
    for f in sorted(root.glob("mqar_*/result.json")):      # aggregate every run under the root (incl. earlier phases)
        r = json.load(open(f))
        if r.get("seed") != seed:
            continue
        key = f"P{r['P']}_D{r['D']}"
        cells.setdefault(key, {"P": r["P"], "D": r["D"], "arms": {}})["arms"][r["arm"]] = {
            "test_acc": r["final_test"]["acc"], "steps_to_90": r["steps_to_90"], "wall": r["wall_seconds"],
            "baselines": r["baselines"], "diverged": r["diverged"]}
    Ps = sorted({c["P"] for c in cells.values()}); Ds = sorted({c["D"] for c in cells.values()})
    arms = [a for a in ARMS if any(a in c["arms"] for c in cells.values())]
    out_ = {"arms": arms, "Ps": Ps, "Ds": Ds, "steps": steps, "seed": seed, "cells": cells, "summary": transition_summary(cells, arms, Ps, Ds)}
    # pilot cell: the cell where the dot_tied/lp_2 mean accuracy is closest to 0.75 (the transition region)
    best, bestd = None, 9
    for k, c in cells.items():
        a = [c["arms"][x]["test_acc"] for x in ("dot_tied", "lp_2") if x in c["arms"]]
        if len(a) == 2 and abs(np.mean(a) - 0.75) < bestd:
            best, bestd = c, abs(np.mean(a) - 0.75)
    out_["pilot_cell"] = {"P": best["P"], "D": best["D"]} if best else None
    print("test accuracy by cell:")
    for D in Ds:
        for P in Ps:
            c = cells.get(f"P{P}_D{D}")
            if c:
                print(f"  D={D} P={P:2d}: " + "  ".join(f"{a}={c['arms'][a]['test_acc']:.3f}" for a in arms if a in c["arms"]))
    print("pilot cell:", out_["pilot_cell"])
    out_["n_keys"] = n_keys
    out_["lr"] = lr_
    return _save(name, out_, out)


# ------------------------------------------------------------------ 8. learning-rate safeguard
def _summarise_trial(run_dir, task):
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


def stage_lr(out=None, quick=False, workers=1, threads=1, base_lr=None, mult=3.0, lm_steps=400, mqar_steps=1000, **_):
    """Each arm at the intended rate and at mult x that rate (truncated runs on the full-length schedule)."""
    lr_lm = base_lr or train.make_cfg("lm", "dot_tied")["lr"]
    lr_mq = base_lr or train.make_cfg("mqar", "dot_tied")["lr"]
    hdr(f"learning-rate safeguard: LM {lr_lm} and {lr_lm * mult:g}; recall {lr_mq} and {lr_mq * mult:g}")
    root = Path(out or PILOT) / "lr_safeguard"
    cell = pilot_cell(out)
    arms = ["dot_tied", "lp_2", "lp_inf"] if quick else ARMS
    cfgs = []
    for k in (1, mult):
        for a in arms:
            lr = lr_lm * k
            cfgs.append(train.make_cfg("lm", a, lr=lr, tag=f"lr{lr:g}", steps=2500, stop_after=lm_steps if not quick else 50, eval_every=50 if quick else 100,
                                       diag_every=50 if quick else 100, ckpt_every=10_000, threads=threads, eval_windows_mid=200))
            lr = lr_mq * k
            cfgs.append(train.make_cfg("mqar", a, lr=lr, tag=f"lr{lr:g}", steps=3000, stop_after=mqar_steps if not quick else 100, eval_every=100, diag_every=100,
                                       ckpt_every=10_000, threads=threads, **_mqar_kw(cell)))
    res = runner.run_many(cfgs, root, workers=workers)
    table = []
    for cfg, r in zip(cfgs, res):
        if "error" in r:
            print("ERROR", r["run_id"], r["error"][-300:]); continue
        s = _summarise_trial(root / train.run_id(cfg), cfg["task"])
        s.update(task=cfg["task"], arm=cfg["arm"], lr=cfg["lr"], diverged=r["diverged"], diverged_step=r["diverged_step"])
        table.append(s)
    out_ = {"lm_lrs": [lr_lm, lr_lm * mult], "mqar_lrs": [lr_mq, lr_mq * mult], "mqar_cell": cell, "lm_steps": lm_steps, "mqar_steps": mqar_steps, "table": table}
    for task in ("lm", "mqar"):
        print(f"--- {task} ---")
        for s in [x for x in table if x["task"] == task]:
            key = "final_val_bpc" if task == "lm" else "final_val_acc"
            print(f"  {s['arm']:10s} lr={s['lr']:.0e} div={s['diverged']} train_loss={s['final_train_loss']:.3f} {key}={s[key]:.3f} "
                  f"max|logit|={s.get('max_abs_logit_max_over_time', float('nan')):.1f} gradnorm_max={s['grad_norm_max']:.2f} "
                  f"clipped={s['frac_log_points_clipped']:.2f} q_norm_growth={s.get('q_norm_growth', float('nan')):.2f}")
    return _save("lr_safeguard", out_, out)


# ------------------------------------------------------------------ 9. seed variance (paired differences)
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


def stage_seedvar(out=None, quick=False, workers=1, threads=1, n_seeds=5, mqar_cells=None, lm_arms=None, lm_steps=None, mqar_steps=3000, **_):
    """Seeds are paired across arms (shared initialisation of common parameters and shared data stream).
    MQAR: several (P, D) cells; LM: the full-length configuration."""
    hdr("seed variance (paired differences)")
    root = Path(out or PILOT) / "seedvar"
    pc = pilot_cell(out)
    if mqar_cells is None:
        mqar_cells = [{"P": pc["P"], "D": pc["D"], "arms": list(ARMS)}]
        if not quick:
            mqar_cells += [{"P": 32, "D": 2, "arms": list(ARMS)}]
    if quick:
        mqar_cells = [{"P": 16, "D": 4, "arms": ["dot_tied", "lp_2"]}]
    lm_arms = lm_arms or (["dot_tied", "lp_2"] if quick else ["dot_tied", "lp_2", "lp_inf"])
    seeds = list(range(100, 100 + (2 if quick else n_seeds)))
    lm_kw = {"steps": lm_steps} if lm_steps else {}
    if quick:
        lm_kw.update(steps=30, eval_every=30, diag_every=30, eval_windows_mid=32)
    cfgs = []
    for cell in mqar_cells:
        for s_ in seeds:
            for a in cell["arms"]:
                cfgs.append(train.make_cfg("mqar", a, seed=s_, steps=mqar_steps if not quick else 200, threads=threads, diag_every=10_000,
                                           eval_every=100, P=cell["P"], D=cell["D"]))
    for s_ in seeds:
        for a in lm_arms:
            cfgs.append(train.make_cfg("lm", a, seed=s_, threads=threads, diag_every=500, eval_every=250, **lm_kw))
    res = runner.run_many(cfgs, root, workers=workers)
    out_ = {"seeds": seeds, "mqar_cells": [{"P": c["P"], "D": c["D"]} for c in mqar_cells], "tasks": {}}
    groups = [(f"mqar_P{c['P']}_D{c['D']}", "mqar", c["arms"], (c["P"], c["D"])) for c in mqar_cells] + [("lm", "lm", lm_arms, None)]
    for name, task, arms, cellkey in groups:
        vals, walls, diverged = {}, {}, {}
        for cfg, r in zip(cfgs, res):
            if cfg["task"] != task or "error" in r or (cellkey and (cfg["P"], cfg["D"]) != cellkey):
                if "error" in r:
                    print("ERROR", r["run_id"], r["error"][-300:])
                continue
            m = r["final_test"]["acc"] if task == "mqar" else r["final_val"]["val_bpc"]
            vals.setdefault(cfg["arm"], []).append(m)
            walls.setdefault(cfg["arm"], []).append(r["wall_seconds"])
            diverged.setdefault(cfg["arm"], []).append(r["diverged"])
        summ = {"metric": "test_acc" if task == "mqar" else "val_bpc", "values": vals, "wall_seconds_mean": {a: float(np.mean(w)) for a, w in walls.items()},
                "diverged": diverged, "single_arm_sd": {a: float(np.std(v, ddof=1)) for a, v in vals.items() if len(v) > 1}, "paired": {}}
        for ctrl in ("dot_tied",):
            for a in arms:
                if a != ctrl and a in vals and ctrl in vals and len(vals[a]) == len(vals[ctrl]) and len(vals[a]) > 1:
                    summ["paired"][a] = paired_summary(vals, ctrl, a)
        out_["tasks"][name] = summ
        print(f"--- {name}: metric {summ['metric']} ---")
        for a in arms:
            if a in vals:
                print(f"  {a:10s} mean={np.mean(vals[a]):.4f} sd={np.std(vals[a], ddof=1):.4f} wall={np.mean(walls[a]):.0f}s")
        for a, p in summ["paired"].items():
            print(f"  paired {a} - dot_tied: mean={p['mean_diff']:+.4f} sd_diff={p['sd_diff']:.4f} corr={p['corr_across_seeds']}")
    return _save("seedvar", out_, out)


# ------------------------------------------------------------------ 10. budget
def _mqar_seconds_table(sv, tr):
    """Seconds per MQAR run (3000 steps, one thread, two processes sharing the machine) for every (cell, arm), measured
    where available and otherwise estimated from the arm ratios seen in the seed-variance cells."""
    table = {}
    for k, c in tr["cells"].items():
        for a, v in c["arms"].items():
            table.setdefault(k, {})[a] = v["wall"]
    ratio = {}
    for name, t in sv["tasks"].items():
        if name.startswith("mqar_") and set(ARMS) <= set(t["wall_seconds_mean"]):
            w = t["wall_seconds_mean"]
            ref = np.mean([w["lp_1"], w["lp_inf"]])
            ratio = {a: w[a] / ref for a in ARMS}
            break
    for k in table:
        have = table[k]
        if ratio and "lp_1" in have and "lp_inf" in have:
            ref = np.mean([have["lp_1"], have["lp_inf"]])
            for a in ARMS:
                have.setdefault(a, ref * ratio[a])
    return table


def project_budget(mqar_table, lm_seconds, cells, mqar_seeds, lm_arms, lm_seeds, factor=1.0, parallel_gain=1.0):
    """Hours of wall-clock time (sum over runs of single-process seconds, divided by the parallel gain)."""
    mq = sum(mqar_table[f"P{c['P']}_D{c['D']}"][a] for c in cells for a in ARMS) * mqar_seeds
    lm = sum(lm_seconds[a] for a in lm_arms) * lm_seeds
    return {"mqar_hours": mq * factor / 3600 / parallel_gain, "lm_hours": lm * factor / 3600 / parallel_gain,
            "total_hours": (mq + lm) * factor / 3600 / parallel_gain}


def stage_budget(out=None, total_hours=48.0, laptop_factor=1.0, **_):
    """Run-time table from the measured pilot runs and a projection of the full plan. laptop_factor scales this
    machine's times to the laptop (re-measure by running the bench and seedvar stages there)."""
    hdr("budget projection")
    sv, tr, bn = _load("seedvar", out), _load("transition", out), _load("bench", out)
    if not (sv and tr):
        print("seedvar/transition results missing"); return None
    res = {"laptop_factor": laptop_factor, "total_hours": total_hours}
    mq = _mqar_seconds_table(sv, tr)
    res["mqar_seconds_per_run"] = mq
    lm = sv["tasks"]["lm"]["wall_seconds_mean"]
    lm_est = dict(lm)
    if bn:
        step = {r["arm"]: r["sec_per_step"] for r in bn["lm_arms"] if "arm" in r and "sec_per_step" in r}
        ref = [a for a in lm if a in step]
        if ref:
            # bench steps were measured with 2 threads; seed-variance runs used one thread, so use the ratio method
            ratio = np.mean([lm[a] / step[a] for a in ref])
            for a in ARMS:
                if a not in lm_est and a in step:
                    lm_est[a] = step[a] * ratio
    res["lm_seconds_per_run"] = lm_est
    pg = bn["parallel"].get("concurrent_over_single", 1.0) if bn else 1.0
    res["note"] = ("seconds are per run in a single-thread process while another run shares the machine; projections divide by 1 "
                   "(the two workers are already included in the measured times)")
    cells = [{"P": P, "D": D} for D in (2, 3, 4, 6) for P in (8, 16, 32, 64)]
    scen = {}
    for lm_seeds in (3, 4, 5):
        if all(a in lm_est for a in ARMS):
            scen[f"mqar16cells_x5seeds_lm{lm_seeds}seeds"] = project_budget(mq, lm_est, cells, 5, ARMS, lm_seeds, laptop_factor, 2.0)
    res["scenarios_hours_two_workers"] = scen
    print("MQAR seconds/run (cells with all arms):")
    for k in sorted(mq):
        if all(a in mq[k] for a in ARMS):
            print(f"  {k}: " + " ".join(f"{a}={mq[k][a]:.0f}" for a in ARMS))
    print("LM seconds/run:", {a: round(v) for a, v in lm_est.items()})
    for k, v in scen.items():
        print(f"  {k}: " + ", ".join(f"{kk}={vv:.1f}h" for kk, vv in v.items()))
    return _save("budget", res, out)


# ------------------------------------------------------------------ 8b. learning-rate scan (full length)
def stage_lr_scan(out=None, quick=False, workers=1, threads=1, lrs=(1e-3, 3e-3, 1e-2), **_):
    """Full-length runs at several rates: is the 'conservative' rate under-training some arm? (A shared rate that
    starves one arm would turn an optimisation artefact into an apparent capacity difference.)"""
    hdr("learning-rate scan (full length)")
    root = Path(out or PILOT) / "lr_scan"
    cfgs = []
    cells = [(64, 3)] if quick else [(64, 3), (32, 4)]
    for lr in lrs:
        for (P, D) in cells:
            for a in (["dot_tied", "lp_2"] if not quick else ["dot_tied"]):
                cfgs.append(train.make_cfg("mqar", a, lr=lr, tag=f"lr{lr:g}", P=P, D=D, steps=3000 if not quick else 200, threads=threads, diag_every=10_000))
    if not quick:
        for lr in (1e-3, 3e-3):       # does more training close the dot-product gap at the hardest cell?
            cfgs.append(train.make_cfg("mqar", "dot_tied", lr=lr, tag=f"lr{lr:g}_long", P=64, D=3, steps=12000, threads=threads, diag_every=10_000, eval_every=500))
        for lr in lrs:
            for a in ("dot_tied", "lp_2", "lp_inf"):
                cfgs.append(train.make_cfg("lm", a, lr=lr, tag=f"lr{lr:g}", threads=threads, diag_every=500, eval_every=250))
    res = runner.run_many(cfgs, root, workers=workers)
    rows = []
    for cfg, r in zip(cfgs, res):
        if "error" in r:
            print("ERROR", r["run_id"], r["error"][-300:]); continue
        row = {"task": cfg["task"], "arm": cfg["arm"], "lr": cfg["lr"], "steps": cfg["steps"], "diverged": r["diverged"], "wall": r["wall_seconds"]}
        if cfg["task"] == "mqar":
            row.update(P=cfg["P"], D=cfg["D"], test_acc=r["final_test"]["acc"])
        else:
            row.update(val_bpc=r["final_val"]["val_bpc"])
        rows.append(row)
        print("  ", json.dumps(row))
    return _save("lr_scan", {"rows": rows}, out)


def stage_transition_ext(out=None, workers=1, threads=1, **kw):
    """Probe beyond the guide's P <= 64 (needs a larger key vocabulary: 1024 keys). Not part of the decided design; it informs
    whether the primary outcome 'P at which accuracy falls below 90%' can be measured for all arms at one D."""
    return stage_transition(out=out, workers=workers, threads=threads, name=_transition_name() + "_ext", n_keys=1024,
                            Ps=[128, 256], Ds=[2, 3], arms=["dot_tied", "lp_2"], **kw)


# ------------------------------------------------------------------ 8c. learning rate per arm (recall task)
def stage_lr_arms(out=None, quick=False, workers=1, threads=1, lrs=(1e-3, 3e-3, 1e-2, 3e-2), cells=((64, 3), (32, 2)), arms=None, seed=0, **_):
    """Arms x several learning rates on the recall task. If one shared rate is near-best for every arm,
    a shared rate is fair; if the best rates differ by arm, a shared rate favours some arms.
    The table aggregates every run found under results/pilot/lr_arms, so the stage can be called again with more rates."""
    hdr("learning rate by arm (recall task)")
    root = Path(out or PILOT) / "lr_arms"
    arms = arms or (["dot_tied", "lp_2"] if quick else ARMS)
    cfgs = []
    for lr in lrs:
        for (P, D) in cells:
            for a in arms:
                cfgs.append(train.make_cfg("mqar", a, lr=lr, tag=f"lr{lr:g}", P=P, D=D, seed=seed, steps=3000 if not quick else 100,
                                           threads=threads, diag_every=10_000))
    res = runner.run_many(cfgs, root, workers=workers)
    for r in res:
        if "error" in r:
            print("ERROR", r["run_id"], r["error"][-300:])
    rows = []
    for f in sorted(root.glob("mqar_*/result.json")):
        r = json.load(open(f))
        c = json.load(open(f.parent / "config.json"))
        if c["seed"] != seed:
            continue
        rows.append({"arm": c["arm"], "lr": c["lr"], "P": c["P"], "D": c["D"], "test_acc": r["final_test"]["acc"],
                     "diverged": r["diverged"], "steps_to_90": r["steps_to_90"], "wall": r["wall_seconds"],
                     "max_abs_logit": max(max(l["max_abs_logit"]) for l in r["layers"]) if r.get("layers") else None})
    all_lrs = sorted({r["lr"] for r in rows})
    all_cells = sorted({(r["P"], r["D"]) for r in rows})
    out_ = {"rows": rows, "lrs": all_lrs, "cells": [list(c) for c in all_cells]}
    for (P, D) in all_cells:
        print(f"--- cell P={P} D={D}: test accuracy by arm and learning rate ---")
        print("  " + "arm".ljust(11) + "".join(f"{lr:>9g}" for lr in all_lrs))
        for a in ARMS:
            v = {r["lr"]: r["test_acc"] for r in rows if r["arm"] == a and r["P"] == P and r["D"] == D}
            if v:
                print("  " + a.ljust(11) + "".join(f"{v.get(lr, float('nan')):9.3f}" for lr in all_lrs))
    return _save("lr_arms", out_, out)