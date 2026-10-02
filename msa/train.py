"""Training loop (both tasks) with checkpoint/resume, logging, diagnostics and final-weight saving.

One run = one (task, arm, seed, config). Everything a run needs is in its config dict, so the
same function is used by the pilot and by the full runs. A run directory contains
  config.json   the full config
  log.jsonl     one JSON line per evaluation / log point
  ckpt.pt       resumable checkpoint (model, optimiser, step, data-stream RNG state), removed at the end
  final.pt      final weights + config + calibration constants + final metrics   (always saved)
  result.json   final metrics, timing, divergence flag
If result.json and final.pt exist the run is skipped; otherwise it resumes from ckpt.pt if present.
"""
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import calibration as cal
from . import evaluate as ev
from . import mqar_gen
from .lm_data import LMData
from .models import CharLM, MQARModel, init_model

MQAR_VAL_SEED, MQAR_TEST_SEED = 777001, 777002

DEFAULTS = {
    "common": dict(seed=0, lr=1e-3, wd=0.01, clip=1.0, warmup=100, min_lr_frac=0.1, target=1.0, r=1e-12,
                   impl="auto", chunk=4, threads=2, log_every=50, ckpt_every=250, diag_every=250),
    "lm": dict(task="lm", d_model=64, n_layers=2, n_heads=4, T=64, mlp_mult=4, batch=64, steps=2500,
               eval_every=250, eval_windows_mid=400),
    "mqar": dict(task="mqar", P=16, Q=None, D=16, d_model=64, n_keys=256, n_vals=64, batch=128, steps=3000,
                 eval_every=100, n_val=5000, n_test=5000, n_val_mid=2000),
}


def make_cfg(task, arm, **kw):
    cfg = copy.deepcopy(DEFAULTS["common"])
    cfg.update(copy.deepcopy(DEFAULTS[task]))
    cfg["arm"] = arm
    cfg.update(kw)
    if task == "mqar" and cfg["Q"] is None:
        cfg["Q"] = cfg["P"] // 2          # decision: Q = P/2
    return cfg


def run_id(cfg, tag=""):
    t = cfg["task"]
    cell = f"P{cfg['P']}_D{cfg['D']}" if t == "mqar" else "lm"
    extra = f"_{cfg['tag']}" if cfg.get("tag") else ""        # cfg['tag'] distinguishes runs that differ only in e.g. the learning rate
    return f"{t}_{cell}_{cfg['arm']}_s{cfg['seed']}{extra}" + (f"_{tag}" if tag else "")


# ------------------------------------------------------------------ construction
_LM_DATA = {}


def get_lm_data(T):
    if T not in _LM_DATA:
        _LM_DATA[T] = LMData(T)
    return _LM_DATA[T]


def build_model(cfg, seed=None):
    seed = cfg["seed"] if seed is None else seed
    if cfg["task"] == "lm":
        data = get_lm_data(cfg["T"])
        m = CharLM(cfg["arm"], data.vocab, cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["T"],
                   cfg["mlp_mult"], cfg["impl"], cfg["chunk"])
    else:
        m = MQARModel(cfg["arm"], cfg["n_keys"], cfg["n_vals"], cfg["d_model"], cfg["D"], cfg["impl"], cfg["chunk"])
    return init_model(m, seed)


def calibration_fns(cfg, n_batches=8):
    """Fixed calibration batches (calibration seed, identical for all arms)."""
    rng = np.random.default_rng(cal.CALIB_SEED)
    fns = []
    if cfg["task"] == "lm":
        data = get_lm_data(cfg["T"])
        for _ in range(n_batches):
            x, _y = data.batch(rng, 32)
            fns.append(lambda m, x=x: m(x))
    else:
        for _ in range(n_batches):
            c, q, _t = mqar_gen.make_batch(rng, 128, cfg["P"], cfg["Q"], cfg["n_keys"], cfg["n_vals"])
            c, q = torch.from_numpy(c), torch.from_numpy(q)
            fns.append(lambda m, c=c, q=q: m(c, q))
    return fns


_CAL_CACHE = {}


def get_constants(cfg):
    """Calibrate on a model initialised with the calibration seed; deterministic and shared across seeds."""
    key = json.dumps({k: cfg[k] for k in ("task", "arm", "target", "r", "impl", "chunk") + (
        ("P", "Q", "D", "d_model", "n_keys", "n_vals") if cfg["task"] == "mqar" else ("T", "d_model", "n_layers", "n_heads", "mlp_mult"))}, sort_keys=True)
    if key not in _CAL_CACHE:
        m = build_model(cfg, seed=cal.CALIB_SEED)
        _CAL_CACHE[key] = cal.calibrate_model(m, calibration_fns(cfg), cfg["target"], cfg["r"])
    return _CAL_CACHE[key]


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if p.dim() == 2 and "emb" not in n:
            decay.append(p)
        else:
            no_decay.append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def lr_at(cfg, step):
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    prog = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
    return cfg["lr"] * (cfg["min_lr_frac"] + (1 - cfg["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * prog)))


# ------------------------------------------------------------------ the run
def _data_rng(cfg):
    return np.random.default_rng(1000 + cfg["seed"])


def _mqar_sets(cfg):
    P, Q, K, V = cfg["P"], cfg["Q"], cfg["n_keys"], cfg["n_vals"]
    val = mqar_gen.fixed_set(MQAR_VAL_SEED + P, cfg["n_val"], P, Q, K, V)
    test = mqar_gen.fixed_set(MQAR_TEST_SEED + P, cfg["n_test"], P, Q, K, V)
    return val, test


def _atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run(cfg, out_root, tag="", verbose=True, stop_at_step=None):
    """Train one run. Returns result dict. `stop_at_step` is for testing interruption."""
    out = Path(out_root) / run_id(cfg, tag)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists() and (out / "final.pt").exists():
        return json.load(open(out / "result.json"))
    torch.set_num_threads(cfg["threads"])
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    consts = get_constants(cfg)
    model = build_model(cfg)
    cal.set_constants(model, consts)
    opt = torch.optim.AdamW(param_groups(model, cfg["wd"]), lr=cfg["lr"], betas=(0.9, 0.99))
    rng = _data_rng(cfg)
    step, wall_prev, log_rows = 0, 0.0, []
    if (out / "ckpt.pt").exists():
        ck = torch.load(out / "ckpt.pt", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        rng.bit_generator.state = ck["rng"]
        step, wall_prev = ck["step"], ck["wall"]
        if (out / "log.jsonl").exists():   # drop log lines written after the checkpoint
            rows = [json.loads(l) for l in open(out / "log.jsonl")]
            log_rows = [r for r in rows if r["step"] <= step]
        if verbose:
            print(f"[resume] {run_id(cfg, tag)} from step {step}", flush=True)
    logf = open(out / "log.jsonl", "w")
    for r in log_rows:
        logf.write(json.dumps(r) + "\n")
    logf.flush()

    task = cfg["task"]
    if task == "lm":
        data = get_lm_data(cfg["T"])
    else:
        val, test = _mqar_sets(cfg)
    t0 = time.time()
    run_loss, run_n, run_gn, max_gn = 0.0, 0, 0.0, 0.0
    diverged, div_step = False, None
    curve = [r for r in log_rows if "val_acc" in r or "val_bpc" in r]

    def log(row):
        row["step"] = step
        row["wall"] = wall_prev + time.time() - t0
        logf.write(json.dumps(row) + "\n")
        logf.flush()

    def diag_batch():
        if task == "lm":
            return data.val_x[:64], data.val_y[:64]
        return val[0][:256], val[1][:256], val[2][:256]

    def diag_fn(m):
        b = diag_batch()
        if task == "lm":
            out_ = m(b[0])
            return F.cross_entropy(out_.reshape(-1, out_.shape[-1]), b[1].reshape(-1))
        c, q, t = (torch.from_numpy(x) for x in b)
        out_ = m(c, q)
        return F.cross_entropy(out_.reshape(-1, out_.shape[-1]), t.reshape(-1))

    def evaluate(final=False):
        row = {}
        if task == "lm":
            r = ev.eval_lm(model, data.val_x, data.val_y, max_windows=None if final else cfg["eval_windows_mid"])
            row.update(val_loss=r["loss"], val_bpc=r["bpc"])
        else:
            n = None if final else cfg["n_val_mid"]
            sl = slice(0, n)
            r = ev.eval_mqar(model, val[0][sl], val[1][sl], val[2][sl], diag=final)
            row.update(val_acc=r["acc"], val_loss=r["loss"])
            if final:
                row["val_diag"] = {k: v for k, v in r.items() if k not in ("acc", "loss")}
        return row

    train_t = 0.0
    end = min(cfg["steps"], cfg.get("stop_after") or cfg["steps"])   # stop_after: truncate a run (LR safeguard trials) while keeping the full-length schedule
    while step < end:
        if stop_at_step is not None and step >= stop_at_step:
            if verbose:
                print(f"[stop] interrupted at step {step} (test)", flush=True)
            logf.close()
            return None
        model.train()
        lr = lr_at(cfg, step)
        for g in opt.param_groups:
            g["lr"] = lr
        ts = time.time()
        if task == "lm":
            x, y = data.batch(rng, cfg["batch"])
            out_ = model(x)
            loss = F.cross_entropy(out_.reshape(-1, out_.shape[-1]), y.reshape(-1))
        else:
            c, q, t = mqar_gen.make_batch(rng, cfg["batch"], cfg["P"], cfg["Q"], cfg["n_keys"], cfg["n_vals"])
            c, q, t = torch.from_numpy(c), torch.from_numpy(q), torch.from_numpy(t)
            out_ = model(c, q)
            loss = F.cross_entropy(out_.reshape(-1, out_.shape[-1]), t.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"]))
        if not (math.isfinite(float(loss)) and math.isfinite(gn)):
            diverged, div_step = True, step
            if verbose:
                print(f"[diverged] {run_id(cfg, tag)} at step {step}: loss={float(loss)} grad_norm={gn}", flush=True)
            break
        opt.step()
        train_t += time.time() - ts
        step += 1
        run_loss += float(loss); run_n += 1; run_gn += gn; max_gn = max(max_gn, gn)
        if step % cfg["log_every"] == 0:
            log({"train_loss": run_loss / run_n, "lr": lr, "grad_norm_mean": run_gn / run_n,
                 "grad_norm_max": max_gn, "sec_per_step": train_t / max(1, step)})
            run_loss, run_n, run_gn, max_gn = 0.0, 0, 0.0, 0.0
        if step % cfg["eval_every"] == 0 or step == end:
            row = evaluate(final=False)
            if step % cfg["diag_every"] == 0 or step == end:
                row["layers"] = ev.collect_scorer_stats(model, diag_fn)
                row["layers_grad"] = ev.collect_scorer_stats(model, diag_fn, with_grad=True)
            log(row)
            curve.append({"step": step, **row})
            if verbose:
                key = "val_bpc" if task == "lm" else "val_acc"
                print(f"  {run_id(cfg, tag)} step {step}/{cfg['steps']} {key}={row[key]:.4f} "
                      f"({wall_prev + time.time() - t0:.0f}s)", flush=True)
        if step % cfg["ckpt_every"] == 0 and step < end:
            _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                          "rng": rng.bit_generator.state, "wall": wall_prev + time.time() - t0}, out / "ckpt.pt")

    wall = wall_prev + time.time() - t0
    result = {"run_id": run_id(cfg, tag), "arm": cfg["arm"], "seed": cfg["seed"], "task": task,
              "steps_done": step, "diverged": diverged, "diverged_step": div_step, "wall_seconds": wall,
              "constants": consts}
    if task == "mqar":
        result["P"], result["D"], result["Q"] = cfg["P"], cfg["D"], cfg["Q"]
    if not diverged:
        fin = evaluate(final=True)
        result["final_val"] = fin
        if task == "mqar":
            result["final_test"] = ev.eval_mqar(model, test[0], test[1], test[2])
            result["sanity"] = ev.mqar_sanity(model, val[0], val[1], val[2])
            result["baselines"] = mqar_gen.baselines(cfg["P"], cfg["n_vals"])
        result["layers"] = ev.collect_scorer_stats(model, diag_fn)
        result["layers_grad"] = ev.collect_scorer_stats(model, diag_fn, with_grad=True)
    key = "val_bpc" if task == "lm" else "val_acc"
    result["curve"] = [{"step": c["step"], key: c[key]} for c in curve if key in c]
    if task == "mqar":   # steps to reach accuracy thresholds (evaluation granularity = eval_every)
        for thr in (0.5, 0.9, 0.99):
            hit = [c["step"] for c in curve if c.get("val_acc", 0) >= thr]
            result[f"steps_to_{int(thr * 100)}"] = hit[0] if hit else None
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "constants": consts, "result_summary":
                {k: v for k, v in result.items() if k not in ("curve", "layers", "layers_grad")}}, out / "final.pt")
    json.dump(result, open(out / "result.json", "w"), indent=1)
    if (out / "ckpt.pt").exists():
        os.remove(out / "ckpt.pt")
    logf.close()
    return result
