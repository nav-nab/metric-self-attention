"""MQAR as a plug-in task for the training engine (msa.train).

Importing this module registers the task "mqar". Configs made with `make_cfg` below carry
`task_module = "shelved.mqar.task"`, so worker processes can import it before running.
"""
import numpy as np
import torch
import torch.nn.functional as F

from msa import train
from . import evaluate as mev
from . import gen
from .model import build as build_mqar

TASK_MODULE = "shelved.mqar.task"
MQAR_VAL_SEED, MQAR_TEST_SEED = 777001, 777002


class MQARTask(train.Task):
    name = "mqar"
    # Defaults at the time of shelving. The learning rate is the common default (1e-3); the pilot found that this under-trains
    # the dot arms and that 3e-2 is much better for them (see PILOT_FINDINGS.md).
    defaults = dict(task="mqar", P=16, Q=None, D=16, d_model=64, n_keys=256, n_vals=64, batch=128, steps=3000,
                    eval_every=100, n_val=5000, n_test=5000, n_val_mid=2000)
    calib_fields = ("P", "Q", "D", "d_model", "n_keys", "n_vals")
    curve_key = "val_acc"

    def finalize_cfg(self, cfg):
        if cfg["Q"] is None:
            cfg["Q"] = cfg["P"] // 2            # decision: Q = P/2

    def cell(self, cfg):
        return f"P{cfg['P']}_D{cfg['D']}"

    def build_model(self, cfg, seed):
        return build_mqar(cfg, seed)

    def calibration_fns(self, cfg, rng, n_batches=8):
        fns = []
        for _ in range(n_batches):
            c, q, _t = gen.make_batch(rng, 128, cfg["P"], cfg["Q"], cfg["n_keys"], cfg["n_vals"])
            c, q = torch.from_numpy(c), torch.from_numpy(q)
            fns.append(lambda m, c=c, q=q: m(c, q))
        return fns

    def setup(self, cfg):
        P, Q, K, V = cfg["P"], cfg["Q"], cfg["n_keys"], cfg["n_vals"]
        val = gen.fixed_set(MQAR_VAL_SEED + P, cfg["n_val"], P, Q, K, V)
        test = gen.fixed_set(MQAR_TEST_SEED + P, cfg["n_test"], P, Q, K, V)
        return {"val": val, "test": test}

    def train_loss(self, state, model, rng, cfg):
        c, q, t = gen.make_batch(rng, cfg["batch"], cfg["P"], cfg["Q"], cfg["n_keys"], cfg["n_vals"])
        c, q, t = torch.from_numpy(c), torch.from_numpy(q), torch.from_numpy(t)
        out = model(c, q)
        return F.cross_entropy(out.reshape(-1, out.shape[-1]), t.reshape(-1))

    def diag_loss(self, state, model):
        val = state["val"]
        c, q, t = (torch.from_numpy(x[:256]) for x in val)
        out = model(c, q)
        return F.cross_entropy(out.reshape(-1, out.shape[-1]), t.reshape(-1))

    def evaluate(self, state, model, cfg, final):
        val = state["val"]
        n = None if final else cfg["n_val_mid"]
        sl = slice(0, n)
        r = mev.eval_mqar(model, val[0][sl], val[1][sl], val[2][sl], diag=final)
        row = {"val_acc": r["acc"], "val_loss": r["loss"]}
        if final:
            row["val_diag"] = {k: v for k, v in r.items() if k not in ("acc", "loss")}
        return row

    def result_header(self, cfg):
        return {"P": cfg["P"], "D": cfg["D"], "Q": cfg["Q"]}

    def finalize(self, state, model, cfg, result):
        test, val = state["test"], state["val"]
        result["final_test"] = mev.eval_mqar(model, test[0], test[1], test[2])
        result["sanity"] = mev.mqar_sanity(model, val[0], val[1], val[2])
        result["baselines"] = gen.baselines(cfg["P"], cfg["n_vals"])

    def finalize_curve(self, result, curve):
        for thr in (0.5, 0.9, 0.99):      # steps to reach accuracy thresholds (granularity = eval_every)
            hit = [c["step"] for c in curve if c.get("val_acc", 0) >= thr]
            result[f"steps_to_{int(thr * 100)}"] = hit[0] if hit else None


train.register_task(MQARTask())


def make_cfg(arm, **kw):
    """Config for one MQAR run (same as msa.train.make_cfg with the task fixed)."""
    return train.make_cfg("mqar", arm, task_module=TASK_MODULE, **kw)
