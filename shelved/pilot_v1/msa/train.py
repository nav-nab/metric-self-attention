"""Training engine: checkpoint/resume, logging, attention diagnostics and final-weight saving, for any registered task.
The language model is the built-in task; further tasks plug in through the Task class below.

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
import importlib
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
from .lm_data import LMData
from .models import CharLM, init_model

DEFAULTS = {
    "common": dict(seed=0, lr=1e-3, wd=0.01, clip=1.0, warmup=100, min_lr_frac=0.1, target=1.0, r=1e-12,
                   impl="auto", chunk=4, threads=2, log_every=50, ckpt_every=250, diag_every=250),
    # qk_dim: query-key width per head; None means d_model / n_heads. The value width per head is always d_model / n_heads.
    "lm": dict(task="lm", d_model=64, n_layers=2, n_heads=4, qk_dim=None, T=64, mlp_mult=4, batch=64, steps=2500,
               eval_every=250, eval_windows_mid=400),
}


# ------------------------------------------------------------------ tasks (plug-ins)
class Task:
    """What the engine needs from a task. The language model is the only built-in task; other tasks (for example the
    shelved MQAR task in mqar) subclass this and call `register_task`. A config selects its task by name;
    configs of an external task also carry `task_module`, so a worker process can import it before building the run."""
    name = None
    defaults = {}               # merged into the config (after DEFAULTS["common"])
    calib_fields = ()           # config fields that determine the calibration constants
    curve_key = None            # metric logged at each evaluation ("val_bpc", "val_acc", ...)

    def finalize_cfg(self, cfg):                 # in-place completion of derived fields
        pass

    def cell(self, cfg):                         # part of the run id
        raise NotImplementedError

    def build_model(self, cfg, seed):            # returns an initialised model (use msa.models.init_model)
        raise NotImplementedError

    def calibration_fns(self, cfg, rng, n_batches=8):    # callables f(model) running one fixed calibration batch
        raise NotImplementedError

    def setup(self, cfg):                        # returns the task's data/state object
        raise NotImplementedError

    def train_loss(self, state, model, rng, cfg):
        raise NotImplementedError

    def diag_loss(self, state, model):           # loss on a fixed small batch, for attention diagnostics
        raise NotImplementedError

    def evaluate(self, state, model, cfg, final):    # returns a row containing curve_key
        raise NotImplementedError

    def result_header(self, cfg):
        return {}

    def finalize(self, state, model, cfg, result):   # extras added to the result after the final evaluation
        pass

    def finalize_curve(self, result, curve):
        pass


TASKS = {}


def register_task(task):
    TASKS[task.name] = task
    DEFAULTS.setdefault(task.name, copy.deepcopy(task.defaults))
    return task


def _import_task(name, module):
    if name not in TASKS and module:
        importlib.import_module(module)
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; external tasks must be imported (config key task_module)")


def get_task(cfg):
    _import_task(cfg["task"], cfg.get("task_module"))
    return TASKS[cfg["task"]]


_LM_DATA = {}


def get_lm_data(T):
    if T not in _LM_DATA:
        _LM_DATA[T] = LMData(T)
    return _LM_DATA[T]


class LMTask(Task):
    name = "lm"
    defaults = DEFAULTS["lm"]
    calib_fields = ("T", "d_model", "n_layers", "n_heads", "mlp_mult", "qk_dim")
    curve_key = "val_bpc"

    def cell(self, cfg):
        return "lm" if not cfg.get("qk_dim") else f"lm_D{cfg['qk_dim']}"

    def build_model(self, cfg, seed):
        data = get_lm_data(cfg["T"])
        m = CharLM(cfg["arm"], data.vocab, cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["T"],
                   cfg["mlp_mult"], cfg["impl"], cfg["chunk"], cfg.get("qk_dim"))
        return init_model(m, seed)

    def calibration_fns(self, cfg, rng, n_batches=8):
        data = get_lm_data(cfg["T"])
        fns = []
        for _ in range(n_batches):
            x, _y = data.batch(rng, 32)
            fns.append(lambda m, x=x: m(x))
        return fns

    def setup(self, cfg):
        return get_lm_data(cfg["T"])

    def train_loss(self, data, model, rng, cfg):
        x, y = data.batch(rng, cfg["batch"])
        out = model(x)
        return F.cross_entropy(out.reshape(-1, out.shape[-1]), y.reshape(-1))

    def diag_loss(self, data, model):
        out = model(data.val_x[:64])
        return F.cross_entropy(out.reshape(-1, out.shape[-1]), data.val_y[:64].reshape(-1))

    def evaluate(self, data, model, cfg, final):
        r = ev.eval_lm(model, data.val_x, data.val_y, max_windows=None if final else cfg["eval_windows_mid"])
        return {"val_loss": r["loss"], "val_bpc": r["bpc"]}


register_task(LMTask())


def make_cfg(task, arm, task_module=None, **kw):
    _import_task(task, task_module)
    cfg = copy.deepcopy(DEFAULTS["common"])
    cfg.update(copy.deepcopy(DEFAULTS[task]))
    cfg["arm"] = arm
    if task_module:
        cfg["task_module"] = task_module
    cfg.update(kw)
    TASKS[task].finalize_cfg(cfg)
    return cfg


def run_id(cfg, tag=""):
    extra = f"_{cfg['tag']}" if cfg.get("tag") else ""        # cfg['tag'] distinguishes runs that differ only in e.g. the learning rate
    return f"{cfg['task']}_{get_task(cfg).cell(cfg)}_{cfg['arm']}_s{cfg['seed']}{extra}" + (f"_{tag}" if tag else "")


# ------------------------------------------------------------------ construction
def build_model(cfg, seed=None):
    """A freshly initialised model (the task builds it and applies the shared per-parameter-name initialisation)."""
    return get_task(cfg).build_model(cfg, cfg["seed"] if seed is None else seed)


def calibration_fns(cfg, n_batches=8):
    """Fixed calibration batches (calibration seed, identical for all arms)."""
    return get_task(cfg).calibration_fns(cfg, np.random.default_rng(cal.CALIB_SEED), n_batches)


_CAL_CACHE = {}


def get_constants(cfg):
    """Calibrate on a model initialised with the calibration seed; deterministic and shared across seeds."""
    task = get_task(cfg)
    key = json.dumps({k: cfg[k] for k in ("task", "arm", "target", "r", "impl", "chunk") + tuple(task.calib_fields)}, sort_keys=True)
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


def _atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run(cfg, out_root, tag="", verbose=True, stop_at_step=None):
    """Train one run. Returns result dict. `stop_at_step` is for testing interruption."""
    task = get_task(cfg)
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

    key = task.curve_key
    state = task.setup(cfg)
    t0 = time.time()
    run_loss, run_n, run_gn, max_gn = 0.0, 0, 0.0, 0.0
    diverged, div_step = False, None
    curve = [r for r in log_rows if key in r]

    def log(row):
        row["step"] = step
        row["wall"] = wall_prev + time.time() - t0
        logf.write(json.dumps(row) + "\n")
        logf.flush()

    def diag_fn(m):
        return task.diag_loss(state, m)

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
        loss = task.train_loss(state, model, rng, cfg)
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
            row = task.evaluate(state, model, cfg, final=False)
            if step % cfg["diag_every"] == 0 or step == end:
                row["layers"] = ev.collect_scorer_stats(model, diag_fn)
                row["layers_grad"] = ev.collect_scorer_stats(model, diag_fn, with_grad=True)
            log(row)
            curve.append({"step": step, **row})
            if verbose:
                print(f"  {run_id(cfg, tag)} step {step}/{cfg['steps']} {key}={row[key]:.4f} "
                      f"({wall_prev + time.time() - t0:.0f}s)", flush=True)
        if step % cfg["ckpt_every"] == 0 and step < end:
            _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                          "rng": rng.bit_generator.state, "wall": wall_prev + time.time() - t0}, out / "ckpt.pt")

    wall = wall_prev + time.time() - t0
    result = {"run_id": run_id(cfg, tag), "arm": cfg["arm"], "seed": cfg["seed"], "task": cfg["task"],
              "steps_done": step, "diverged": diverged, "diverged_step": div_step, "wall_seconds": wall,
              "constants": consts}
    result.update(task.result_header(cfg))
    if not diverged:
        result["final_val"] = task.evaluate(state, model, cfg, final=True)
        task.finalize(state, model, cfg, result)
        result["layers"] = ev.collect_scorer_stats(model, diag_fn)
        result["layers_grad"] = ev.collect_scorer_stats(model, diag_fn, with_grad=True)
    result["curve"] = [{"step": c["step"], key: c[key]} for c in curve if key in c]
    task.finalize_curve(result, curve)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "constants": consts, "result_summary":
                {k: v for k, v in result.items() if k not in ("curve", "layers", "layers_grad")}}, out / "final.pt")
    json.dump(result, open(out / "result.json", "w"), indent=1)
    if (out / "ckpt.pt").exists():
        os.remove(out / "ckpt.pt")
    logf.close()
    return result
