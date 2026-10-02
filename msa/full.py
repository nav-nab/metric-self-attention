"""Full experiment: expands configs/full.json into runs and executes them with resume support.

    python execute_project.py full --config configs/full.json --workers 2 [--status]

Re-issuing the command continues an interrupted job: finished runs are skipped, and a run that was
interrupted mid-training resumes from its last checkpoint (bit-exact). Final weights of every run are in
results/full/<run_id>/final.pt.
"""
import json
from pathlib import Path

from . import runner, train

ROOT = Path(__file__).resolve().parent.parent


def expand(conf):
    s = conf.get("settings", {})
    cfgs = []
    m = conf.get("mqar")
    if m:
        for cell in m["cells"]:
            for seed in m["seeds"]:
                for arm in m["arms"]:
                    cfgs.append(train.make_cfg("mqar", arm, seed=seed, P=cell["P"], D=cell["D"], steps=m.get("steps", 3000),
                                               diag_every=m.get("diag_every", 1000), **s, **m.get("overrides", {})))
    l = conf.get("lm")
    if l:
        for seed in l["seeds"]:
            for arm in l["arms"]:
                cfgs.append(train.make_cfg("lm", arm, seed=seed, **{**s, **l.get("overrides", {})}))
    return cfgs


def status(cfgs, out_root):
    done = [c for c in cfgs if (Path(out_root) / train.run_id(c) / "result.json").exists()]
    part = [c for c in cfgs if (Path(out_root) / train.run_id(c) / "ckpt.pt").exists()]
    return len(done), len(part), len(cfgs)


def run_full(config_path, workers=1, threads=None, out=None, only_status=False):
    conf = json.load(open(config_path))
    out = Path(out or ROOT / "results" / "full")
    cfgs = expand(conf)
    if threads:
        for c in cfgs:
            c["threads"] = threads
    d, p, n = status(cfgs, out)
    print(f"{n} runs in manifest: {d} finished, {p} partially trained (will resume), {n - d - p} not started")
    if only_status:
        return
    todo = [c for c in cfgs if not (out / train.run_id(c) / "result.json").exists()]
    runner.run_many(todo, out, workers=workers)
    d, p, n = status(cfgs, out)
    print(f"finished {d}/{n}")
