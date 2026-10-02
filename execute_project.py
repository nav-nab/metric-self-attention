#!/usr/bin/env python
"""Single entry point for the whole project (pilot checks, benchmarks, and the full runs).

Pilot stages (each writes results/pilot/<stage>.json):
  python execute_project.py env            environment check + smoke training step
  python execute_project.py gradcheck      float64 gradient check, custom kernel vs naive autograd
  python execute_project.py generator      MQAR generator checks, baselines, sanity checks
  python execute_project.py bench          timing and memory per arm at real model sizes (both tasks)
  python execute_project.py calibrate      calibration constants, typical sums inside the root
  python execute_project.py transition     recall transition scan (accuracy vs P, D per arm)
  python execute_project.py lr             learning-rate safeguard (base rate and 3x, truncated runs)
  python execute_project.py epsilon        choice of the relative epsilon r (uses the trained models from the lr stage)
  python execute_project.py lr_scan        full-length runs at several learning rates
  python execute_project.py lr_arms        all arms x several learning rates on the recall task
  python execute_project.py transition_ext probe beyond P=64 (1024 keys) for dot_tied and lp_2 (informs the primary outcome)
  python execute_project.py seedvar        seed variance, paired differences
  python execute_project.py budget         run-time table and budget projection
  python execute_project.py pilot          all of the above in order
Full experiment (resumable):
  python execute_project.py full --config configs/full.json --workers 2 [--status]
Tests:
  python execute_project.py tests
Options: --quick (tiny smoke version of a stage), --workers N (parallel processes), --threads N (threads per process),
         --out DIR (results directory).
"""
import argparse
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["env", "gradcheck", "generator", "bench", "calibrate", "transition", "epsilon", "lr", "lr_scan", "lr_arms", "transition_ext", "seedvar",
                                      "budget", "pilot", "full", "tests"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default="configs/full.json")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--laptop-factor", type=float, default=1.0)
    ap.add_argument("--time-budget", type=float, default=None, help="bench only: stop after this many seconds; re-run to continue")
    ap.add_argument("--lr-mqar", type=float, default=None, help="override the recall-task learning rate for this invocation")
    ap.add_argument("--lr-lm", type=float, default=None, help="override the language-model learning rate for this invocation")
    ap.add_argument("--r", type=float, default=None, help="override the relative epsilon r")
    ap.add_argument("--lrs", default=None, help="lr_arms: comma-separated learning rates")
    ap.add_argument("--arms", default=None, help="comma-separated arm list (transition, lr_arms)")
    ap.add_argument("--Ps", default=None, help="transition: comma-separated pair counts")
    ap.add_argument("--Ds", default=None, help="transition: comma-separated head dimensions")
    ap.add_argument("--fresh", action="store_true", help="bench only: ignore cached measurements")
    a = ap.parse_args()
    if a.stage == "tests":
        import pytest
        sys.exit(pytest.main(["-q", "tests"]))
    from msa import pilot, full
    if a.stage == "full":
        full.run_full(a.config, workers=a.workers, threads=a.threads or 1, out=a.out, only_status=a.status)
        return
    kw = dict(out=a.out, quick=a.quick, workers=a.workers)
    if a.threads:
        kw["threads"] = a.threads
    from msa import train
    if a.lr_mqar:
        train.DEFAULTS["mqar"]["lr"] = a.lr_mqar
    if a.lr_lm:
        train.DEFAULTS["lm"]["lr"] = a.lr_lm
    if a.r is not None:
        train.DEFAULTS["common"]["r"] = a.r
    if a.lrs:
        kw["lrs"] = tuple(float(x) for x in a.lrs.split(","))
    if a.arms:
        kw["arms"] = a.arms.split(",")
    if a.Ps:
        kw["Ps"] = [int(x) for x in a.Ps.split(",")]
    if a.Ds:
        kw["Ds"] = [int(x) for x in a.Ds.split(",")]
    if a.stage == "bench":
        kw.update(time_budget=a.time_budget, fresh=a.fresh)
    stages = {"env": pilot.stage_env, "gradcheck": pilot.stage_gradcheck, "generator": pilot.stage_generator,
              "bench": pilot.stage_bench, "calibrate": pilot.stage_calibrate, "transition": pilot.stage_transition,
              "epsilon": pilot.stage_epsilon, "lr": pilot.stage_lr, "lr_scan": pilot.stage_lr_scan, "lr_arms": pilot.stage_lr_arms, "transition_ext": pilot.stage_transition_ext, "seedvar": pilot.stage_seedvar,
              "budget": lambda **k: pilot.stage_budget(laptop_factor=a.laptop_factor, **k)}
    order = ["env", "gradcheck", "generator", "bench", "calibrate", "transition", "lr", "epsilon", "lr_scan", "seedvar", "budget"]
    todo = order if a.stage == "pilot" else [a.stage]
    for s in todo:
        t0 = time.time()
        stages[s](**kw)
        print(f"[{s}] finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
