"""Full experiment: expands a JSON config into language-model runs and executes them with resume support.

    python execute_project.py full --config configs/full.json --workers 2 [--status]

Config format (no config file is shipped yet: the design is still being settled with the user):

    {"settings": {"lr": 3e-3},                         # optional overrides applied to every run (any make_cfg field)
     "lm": {"seeds": [0, 1, 2, 3, 4],
            "arms": ["dot_tied", "lp_1", "lp_2", "lp_inf", "dot_untied"],
            "qk_dims": [2, 4, 16],                     # head sizes (query-key width per head); null/absent = d_model / n_heads
            "overrides": {},                           # optional per-block overrides
            "uniform": {"seeds": [0, 1, 2, 3, 4]}}}    # optional: the uniform-attention control (independent of head size)

Re-issuing the command continues an interrupted job: finished runs are skipped, and a run that was
interrupted mid-training resumes from its last checkpoint (bit-exact). Final weights of every run are in
results/full/<run_id>/final.pt.
"""
import json
from pathlib import Path

from . import runner, train
from .arms import NULL_ARM

ROOT = Path(__file__).resolve().parent.parent


def expand(conf):
    s = conf.get("settings", {})
    cfgs = []
    l = conf.get("lm")
    if l:
        over = {**s, **l.get("overrides", {})}
        for qk in l.get("qk_dims") or [None]:
            for seed in l["seeds"]:
                for arm in l["arms"]:
                    cfgs.append(train.make_cfg("lm", arm, seed=seed, **({"qk_dim": qk} if qk else {}), **over))
        u = l.get("uniform")
        if u:
            for seed in u["seeds"]:
                cfgs.append(train.make_cfg("lm", NULL_ARM, seed=seed, **over))
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
