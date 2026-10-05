#!/usr/bin/env python
"""Single entry point for the project (language-model pilot checks, benchmarks, and the full runs).

Stages (each writes results/pilot/<stage>.json):
  python execute_project.py env            environment check + smoke training steps (incl. a small head and the uniform control)
  python execute_project.py gradcheck      float64 gradient check, custom kernel vs naive autograd
  python execute_project.py bench          timing and memory per arm (head size 16 and small head sizes)
  python execute_project.py calibrate      calibration constants and typical sums inside the root, per head size
  python execute_project.py epsilon        gradient safety of the relative epsilon r
  python execute_project.py lr             learning-rate safeguard (base rate and 3x, truncated runs)
  python execute_project.py lr_scan        full-length runs at several learning rates (head size 16)
  python execute_project.py seedvar        seed variance, paired differences (head size 16)
  python execute_project.py lm_small_d_lr  small query-key dimension: rate scan and uniform control
  python execute_project.py lm_small_d     small query-key dimension: all arms, paired seeds (use --lr-lm, --Ds, --seeds)
  python execute_project.py budget         run-time table and budget projection
  python execute_project.py pilot          env, gradcheck, bench, calibrate, epsilon, lr, lr_scan, seedvar, budget in order
Full experiment (resumable; the config format is documented in msa/full.py):
  python execute_project.py full --config configs/full.json --workers 2 [--status]
Tests:
  python execute_project.py tests
Options: --quick (tiny smoke version of a stage), --workers N (parallel processes), --threads N (threads per process),
         --out DIR (results directory).
The recall task (MQAR) is shelved: see mqar/README.md and mqar/execute_mqar.py.
"""
import argparse
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["env", "gradcheck", "bench", "calibrate", "epsilon", "lr", "lr_scan", "seedvar", "lm_small_d_lr", "lm_small_d",
                                      "budget", "pilot", "full", "tests"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default="configs/full.json")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--laptop-factor", type=float, default=1.0)
    ap.add_argument("--time-budget", type=float, default=None, help="bench only: stop after this many seconds; re-run to continue")
    ap.add_argument("--lr-lm", type=float, default=None, help="override the language-model learning rate for this invocation")
    ap.add_argument("--r", type=float, default=None, help="override the relative epsilon r")
    ap.add_argument("--lrs", default=None, help="comma-separated learning rates (lr_scan, lm_small_d_lr)")
    ap.add_argument("--arms", default=None, help="comma-separated arm list (lm_small_d)")
    ap.add_argument("--Ds", default=None, help="comma-separated head dimensions (calibrate, lm_small_d, lm_small_d_lr)")
    ap.add_argument("--seeds", type=int, default=None, help="number of paired seeds (seedvar, lm_small_d)")
    ap.add_argument("--fresh", action="store_true", help="bench only: ignore cached measurements")
    a = ap.parse_args()
    if a.stage == "tests":
        import pytest
        root = Path(__file__).resolve().parent
        sys.exit(pytest.main(["-q", str(root / "tests"), str(root / "mqar")]))
    from msa import pilot, full
    if a.stage == "full":
        full.run_full(a.config, workers=a.workers, threads=a.threads or 1, out=a.out, only_status=a.status)
        return
    kw = dict(out=a.out, quick=a.quick, workers=a.workers)
    if a.threads:
        kw["threads"] = a.threads
    from msa import train
    if a.lr_lm:
        train.DEFAULTS["lm"]["lr"] = a.lr_lm
    if a.r is not None:
        train.DEFAULTS["common"]["r"] = a.r
    if a.lrs:
        kw["lrs"] = tuple(float(x) for x in a.lrs.split(","))
    if a.arms:
        kw["arms"] = a.arms.split(",")
    if a.Ds:
        kw["Ds"] = [int(x) for x in a.Ds.split(",")]
    if a.seeds:
        kw["n_seeds"] = a.seeds
    if a.stage == "bench":
        kw.update(time_budget=a.time_budget, fresh=a.fresh)
    stages = {"env": pilot.stage_env, "gradcheck": pilot.stage_gradcheck, "bench": pilot.stage_bench,
              "calibrate": pilot.stage_calibrate, "epsilon": pilot.stage_epsilon, "lr": pilot.stage_lr,
              "lr_scan": pilot.stage_lr_scan, "seedvar": pilot.stage_seedvar, "lm_small_d_lr": pilot.stage_lm_small_d_lr,
              "lm_small_d": pilot.stage_lm_small_d,
              "budget": lambda **k: pilot.stage_budget(laptop_factor=a.laptop_factor, **k)}
    order = ["env", "gradcheck", "bench", "calibrate", "epsilon", "lr", "lr_scan", "seedvar", "budget"]
    todo = order if a.stage == "pilot" else [a.stage]
    for s in todo:
        t0 = time.time()
        stages[s](**kw)
        print(f"[{s}] finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
