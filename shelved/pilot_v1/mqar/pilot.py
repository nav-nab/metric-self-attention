"""MQAR pilot stages (SHELVED). These are the recall-task stages of the original pilot, re-pointed at the plug-in task
(mqar/task.py). Results go to results/mqar/ by default; the original pilot outputs stay in results/pilot/.

Run through `python mqar/execute_mqar.py <stage>`; see README.md in this folder.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from msa import lp, runner, train
from msa.arms import ARMS
from msa import pilot as _main_pilot
from msa.pilot import _summarise_trial, hdr, paired_summary  # shared helpers

from . import evaluate as mev
from . import gen
from .model import build as build_mqar
from .task import MQAR_VAL_SEED, make_cfg

ROOT = Path(__file__).resolve().parents[1]
MQAR_OUT = ROOT / "results" / "mqar"


def _save(name, obj, out=None):
    return _main_pilot._save(name, obj, out or MQAR_OUT)


def _load(name, out=None):
    return _main_pilot._load(name, out or MQAR_OUT)


def stage_generator(out=None, **_):
    hdr("MQAR generator check")
    res = {"cells": [], "ok": True}
    for P in (8, 16, 32, 64):
        r = gen.check_generator(P=P, Q=P // 2, n_keys=256, n_vals=64, n=20000, seed=P)
        res["cells"].append(r)
        res["ok"] &= r["ok"]
        v = r["vectorised"]
        print(f"P={P:2d} Q={P//2:2d}: ok={r['ok']} chance sim/theory={v['chance_sim']:.4f}/{v['chance_theory']:.4f} "
              f"key-blind sim/theory={v['key_blind_sim']:.4f}/{v['key_blind_theory']:.4f}")
    # equivalence of the fixed evaluation sets across arms: same seed -> identical arrays
    a = gen.fixed_set(777001 + 16, 100, 16, 8, 256, 64)
    b = gen.fixed_set(777001 + 16, 100, 16, 8, 256, 64)
    res["fixed_sets_identical"] = all((x == y).all() for x, y in zip(a, b))
    # model-level sanity on untrained models (the trained-model versions are recorded in every run)
    ctx, qry, tgt = gen.fixed_set(1, 2000, 16, 8, 256, 64)
    sanity = {}
    for arm in ("dot_tied", "lp_2"):
        m = build_mqar({**make_cfg(arm)}, 0)
        sanity[arm] = mev.mqar_sanity(m, ctx, qry, tgt)
    res["untrained_model_sanity"] = sanity
    res["shuffle_invariant"] = all(s["shuffle_abs_diff"] < 1e-9 for s in sanity.values())
    res["no_context_at_chance"] = all(abs(s["acc_no_context"] - s["chance"]) < 0.01 for s in sanity.values())
    res["ok"] &= res["fixed_sets_identical"] and res["shuffle_invariant"] and res["no_context_at_chance"]
    print(f"fixed sets identical: {res['fixed_sets_identical']}; shuffle invariant: {res['shuffle_invariant']}; "
          f"no-context at chance: {res['no_context_at_chance']}")
    return _save("generator", res, out)


def _mqar_kw(cell):
    return {"P": cell["P"], "D": cell["D"]}


def _transition_name():
    """Transition results are kept per learning rate: 'transition' for the original 1e-3, 'transition_lr<lr>' otherwise."""
    lr = make_cfg("dot_tied")["lr"]
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


def stage_transition(out=None, quick=False, workers=1, threads=1, steps=3000, arms=None, Ps=None, Ds=None, seed=0, name=None, n_keys=256, **_):
    """One seed per (arm, P, D) cell: where does accuracy start to drop, for each arm?"""
    name = name or _transition_name()
    hdr(f"recall transition scan ({name})")
    arms = arms or (["dot_tied", "lp_2"] if quick else ["dot_tied", "lp_1", "lp_2", "lp_inf", "dot_untied"])
    Ps = Ps or ([16, 32] if quick else [8, 16, 32, 64])
    Ds = Ds or ([2, 4] if quick else [2, 3, 4, 6, 8])
    lr_ = make_cfg("dot_tied")["lr"]
    root = Path(out or MQAR_OUT) / name
    cfgs = [make_cfg(a, P=P, D=D, steps=steps, seed=seed, threads=threads, diag_every=steps, n_keys=n_keys)
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


def stage_transition_ext(out=None, workers=1, threads=1, **kw):
    """Probe beyond the guide's P <= 64 (needs a larger key vocabulary: 1024 keys). Not part of the decided design; it informs
    whether the primary outcome 'P at which accuracy falls below 90%' can be measured for all arms at one D."""
    return stage_transition(out=out, workers=workers, threads=threads, name=_transition_name() + "_ext", n_keys=1024,
                            Ps=[128, 256], Ds=[2, 3], arms=["dot_tied", "lp_2"], **kw)


def stage_lr_arms(out=None, quick=False, workers=1, threads=1, lrs=(1e-3, 3e-3, 1e-2, 3e-2), cells=((64, 3), (32, 2)), arms=None, seed=0, **_):
    """Arms x several learning rates on the recall task. If one shared rate is near-best for every arm,
    a shared rate is fair; if the best rates differ by arm, a shared rate favours some arms.
    The table aggregates every run found under results/pilot/lr_arms, so the stage can be called again with more rates."""
    hdr("learning rate by arm (recall task)")
    root = Path(out or MQAR_OUT) / "lr_arms"
    arms = arms or (["dot_tied", "lp_2"] if quick else ARMS)
    cfgs = []
    for lr in lrs:
        for (P, D) in cells:
            for a in arms:
                cfgs.append(make_cfg(a, lr=lr, tag=f"lr{lr:g}", P=P, D=D, seed=seed, steps=3000 if not quick else 100,
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


# ------------------------------------------------------------------ learning-rate safeguard (recall half)
def stage_lr_safeguard(out=None, quick=False, workers=1, threads=1, base_lr=None, mult=3.0, steps=1000, arms=None, **_):
    """Each arm at the intended rate and at mult x that rate (truncated runs on the full-length schedule)."""
    lr0 = base_lr or make_cfg("dot_tied")["lr"]
    hdr(f"MQAR learning-rate safeguard: {lr0} and {lr0 * mult:g}")
    root = Path(out or MQAR_OUT) / "lr_safeguard"
    cell = pilot_cell(out)
    arms = arms or (["dot_tied", "lp_2", "lp_inf"] if quick else ARMS)
    cfgs = []
    for k in (1, mult):
        for a in arms:
            lr = lr0 * k
            cfgs.append(make_cfg(a, lr=lr, tag=f"lr{lr:g}", steps=3000, stop_after=steps if not quick else 100, eval_every=100,
                                 diag_every=100, ckpt_every=10_000, threads=threads, **_mqar_kw(cell)))
    res = runner.run_many(cfgs, root, workers=workers)
    table = []
    for cfg, r in zip(cfgs, res):
        if "error" in r:
            print("ERROR", r["run_id"], r["error"][-300:])
            continue
        s = _summarise_trial(root / train.run_id(cfg), "mqar")
        s.update(arm=cfg["arm"], lr=cfg["lr"], diverged=r["diverged"], diverged_step=r["diverged_step"])
        table.append(s)
        print(f"  {s['arm']:10s} lr={s['lr']:.0e} div={s['diverged']} train_loss={s['final_train_loss']:.3f} "
              f"val_acc={s['final_val_acc']:.3f} gradnorm_max={s['grad_norm_max']:.2f}")
    return _save("lr_safeguard", {"lrs": [lr0, lr0 * mult], "cell": cell, "steps": steps, "table": table}, out)


# ------------------------------------------------------------------ epsilon: training sensitivity and trained scale
def stage_epsilon_runs(out=None, quick=False, workers=1, threads=1, **_):
    """Does the relative epsilon r (and the p = 2 implementation path) change the outcome at the pilot cell?
    Also records the scale of the sums inside the root that trained models produce for matched pairs (read from the
    models of the lr_safeguard stage when they exist). The gradient-safety part of the epsilon question is in
    msa/pilot.py (stage epsilon) and is task-independent."""
    hdr("MQAR epsilon: training sensitivity and trained scale")
    res = {"trained": [], "training_sensitivity": []}
    root = Path(out or MQAR_OUT) / "lr_safeguard"
    if root.exists():
        for f in sorted(root.glob("mqar_*_lp_*_lr0.001/final.pt")):
            ck = torch.load(f, weights_only=False)
            cfg = ck["cfg"]
            if cfg["arm"] == "lp_inf":
                continue
            m = build_mqar(cfg, cfg["seed"])
            m.load_state_dict(ck["state_dict"])
            p = m.scorer.p
            P, Q, K, V = cfg["P"], cfg["Q"], cfg["n_keys"], cfg["n_vals"]
            val = gen.fixed_set(MQAR_VAL_SEED + P, cfg["n_val"], P, Q, K, V)
            c_, q_, _t = (torch.from_numpy(x[:500]) for x in val)
            with torch.no_grad():
                _, att, logits, zk = m(c_, q_, return_att=True)
                xq = m.key_emb(q_) + m.val_emb.weight[m.n_vals]
                zq = m.W_q(m.ln_in(xq))
                S = lp.power_sum(zq.unsqueeze(1), zk.unsqueeze(1), p)[:, 0]
                match = (c_[:, None, :, 0] == q_[:, :, None])
                S_match, S_rest = S[match], S[~match]
            S_typ0 = train.get_constants(cfg)[0]["S_typ"]
            res["trained"].append({"arm": cfg["arm"], "S_typ_init": S_typ0, "S_match_median": float(S_match.median()),
                                   "S_match_q01": float(torch.quantile(S_match, 0.01)), "S_match_min": float(S_match.min()),
                                   "S_other_median": float(S_rest.median()),
                                   "match_over_Styp_init_q01": float(torch.quantile(S_match, 0.01)) / S_typ0})
            print(f"  trained {cfg['arm']:7s}: S_typ(init)={S_typ0:.3g} matched-pair S median={float(S_match.median()):.3g}")
    cell = pilot_cell(out)
    eroot = Path(out or MQAR_OUT) / "epsilon_runs"
    seeds = (0,) if quick else (0, 1)
    nsteps = 100 if quick else 3000
    cfgs = []
    for arm in (["lp_8"] if quick else ["lp_2", "lp_4", "lp_8"]):
        for r_ in (1e-12, 1e-6, 1e-3):
            for sd in seeds:
                cfgs.append(make_cfg(arm, r=r_, tag=f"r{r_:g}", seed=sd, steps=nsteps, threads=threads, diag_every=10_000, **_mqar_kw(cell)))
    for sd in seeds:        # p = 2: matmul expansion vs direct differences (custom kernel), same r
        cfgs.append(make_cfg("lp_2", r=1e-12, impl="custom", tag="r1e-12_direct", seed=sd, steps=nsteps, threads=threads,
                             diag_every=10_000, **_mqar_kw(cell)))
    results = runner.run_many(cfgs, eroot, workers=workers)
    for cfg, r_ in zip(cfgs, results):
        if "error" in r_:
            print("ERROR", r_["run_id"], r_["error"][-300:])
            continue
        row = {"arm": cfg["arm"], "r": cfg["r"], "impl": cfg["impl"], "seed": cfg["seed"], "test_acc": r_["final_test"]["acc"],
               "steps_to_90": r_["steps_to_90"], "wall": r_["wall_seconds"]}
        res["training_sensitivity"].append(row)
        print("  ", json.dumps(row))
    res["cell"] = cell
    return _save("epsilon_runs", res, out)


# ------------------------------------------------------------------ seed variance (recall half)
def stage_seedvar(out=None, quick=False, workers=1, threads=1, n_seeds=5, cells=None, steps=3000, **_):
    """Paired seeds across arms at the pilot cell (and a second cell with D = 2): seed-to-seed spread and detectable differences."""
    hdr("MQAR seed variance (paired differences)")
    root = Path(out or MQAR_OUT) / "seedvar"
    pc = pilot_cell(out)
    if cells is None:
        cells = [{"P": pc["P"], "D": pc["D"], "arms": list(ARMS)}] + ([] if quick else [{"P": 32, "D": 2, "arms": list(ARMS)}])
    if quick:
        cells = [{"P": 16, "D": 4, "arms": ["dot_tied", "lp_2"]}]
    seeds = list(range(100, 100 + (2 if quick else n_seeds)))
    cfgs = [make_cfg(a, seed=s_, steps=steps if not quick else 200, threads=threads, diag_every=10_000, eval_every=100,
                     P=c["P"], D=c["D"]) for c in cells for s_ in seeds for a in c["arms"]]
    res = runner.run_many(cfgs, root, workers=workers)
    out_ = {"seeds": seeds, "cells": [{"P": c["P"], "D": c["D"]} for c in cells], "tasks": {}}
    for c in cells:
        name = f"mqar_P{c['P']}_D{c['D']}"
        vals, walls = {}, {}
        for cfg, r in zip(cfgs, res):
            if "error" in r:
                print("ERROR", r["run_id"], r["error"][-300:])
                continue
            if (cfg["P"], cfg["D"]) != (c["P"], c["D"]):
                continue
            vals.setdefault(cfg["arm"], []).append(r["final_test"]["acc"])
            walls.setdefault(cfg["arm"], []).append(r["wall_seconds"])
        summ = {"metric": "test_acc", "values": vals, "wall_seconds_mean": {a: float(np.mean(w)) for a, w in walls.items()},
                "single_arm_sd": {a: float(np.std(v, ddof=1)) for a, v in vals.items() if len(v) > 1}, "paired": {}}
        for a in c["arms"]:
            if a != "dot_tied" and a in vals and "dot_tied" in vals and len(vals[a]) == len(vals["dot_tied"]) > 1:
                summ["paired"][a] = paired_summary(vals, "dot_tied", a)
        out_["tasks"][name] = summ
        print(f"--- {name} ---")
        for a in c["arms"]:
            if a in vals:
                print(f"  {a:10s} mean={np.mean(vals[a]):.4f} sd={np.std(vals[a], ddof=1):.4f}")
    return _save("seedvar", out_, out)
