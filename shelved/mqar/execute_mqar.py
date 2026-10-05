#!/usr/bin/env python
"""Entry point for the SHELVED MQAR (multi-query associative recall) task. Nothing here runs unless you call it.

  python shelved/mqar/execute_mqar.py generator        generator checks, baselines, sanity checks
  python shelved/mqar/execute_mqar.py transition       accuracy vs (P, D) per arm
  python shelved/mqar/execute_mqar.py transition_ext   probe beyond P = 64 (1024 keys)
  python shelved/mqar/execute_mqar.py lr_arms          all arms x several learning rates
  python shelved/mqar/execute_mqar.py lr_safeguard     base rate and 3x, truncated runs
  python shelved/mqar/execute_mqar.py epsilon_runs     training sensitivity to the relative epsilon r
  python shelved/mqar/execute_mqar.py seedvar          paired seeds at the pilot cell
  python shelved/mqar/execute_mqar.py tests            the shelved tests
Options: --quick, --workers N, --threads N, --out DIR (default results/mqar), --lr, --Ps, --Ds, --arms, --lrs, --seeds.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # repo root, so that `msa` and `shelved` import


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["generator", "transition", "transition_ext", "lr_arms", "lr_safeguard", "epsilon_runs", "seedvar", "tests"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lr", type=float, default=None, help="override the recall learning rate (default 1e-3)")
    ap.add_argument("--lrs", default=None)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--Ps", default=None)
    ap.add_argument("--Ds", default=None)
    ap.add_argument("--seeds", type=int, default=None)
    a = ap.parse_args()
    if a.stage == "tests":
        import pytest
        sys.exit(pytest.main(["-q", str(Path(__file__).resolve().parent / "tests")]))
    from msa import train
    from shelved.mqar import pilot, task            # noqa: F401  (importing task registers "mqar")
    if a.lr:
        task.MQARTask.defaults["lr"] = a.lr
    kw = dict(out=a.out, quick=a.quick, workers=a.workers)
    if a.threads:
        kw["threads"] = a.threads
    if a.lrs:
        kw["lrs"] = tuple(float(x) for x in a.lrs.split(","))
    if a.arms:
        kw["arms"] = a.arms.split(",")
    if a.Ps:
        kw["Ps"] = [int(x) for x in a.Ps.split(",")]
    if a.Ds:
        kw["Ds"] = [int(x) for x in a.Ds.split(",")]
    if a.seeds:
        kw["n_seeds"] = a.seeds
    stage = {"generator": pilot.stage_generator, "transition": pilot.stage_transition, "transition_ext": pilot.stage_transition_ext,
             "lr_arms": pilot.stage_lr_arms, "lr_safeguard": pilot.stage_lr_safeguard, "epsilon_runs": pilot.stage_epsilon_runs,
             "seedvar": pilot.stage_seedvar}[a.stage]
    t0 = time.time()
    stage(**kw)
    print(f"[{a.stage}] finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
