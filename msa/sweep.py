"""The stages that run many runs: the learning-rate scan, the rate selection and the main grid.

All state is in files. A run is finished when its result.json exists, so every stage can be
interrupted and started again. About four worker processes run at once, one thread each, and
the slowest cells go first so that the last runs of a stage are short.
"""
import csv
import io
import json
import multiprocessing
import signal
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from .calibrate import run_calibration
from .data import FINAL_SEEDS, SCAN_SEED
from .scores import CROSSED_ARMS, HEAD_SIZES
from .train import CKPT_SECONDS, EVAL_EVERY, LOG_EVERY, STEPS, WARMUP, RunSpec, run_training
from .util import atomic_write_json, atomic_write_text

HEARTBEAT_SECONDS = 60        # a status line about once a minute while runs are going
SCAN_RATES = (3e-3, 1e-2, 3e-2)
EXTRA_RATE_LOW = 1e-3         # one more rate, a factor of 3 beyond the grid
EXTRA_RATE_HIGH = 9e-2


class SweepError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    """Everything that a smoke run shrinks. The defaults are the real experiment."""
    steps: int = STEPS
    warmup: int = WARMUP
    eval_every: int = EVAL_EVERY
    log_every: int = LOG_EVERY
    final_seeds: tuple = FINAL_SEEDS
    scan_seed: int = SCAN_SEED
    workers: int = 4
    ckpt_seconds: float = CKPT_SECONDS
    calibration_batches: int = 32


@dataclass(frozen=True)
class Cell:
    arm: str
    head_dim: int

    @property
    def id(self):
        return self.arm if self.arm == "uniform" else f"{self.arm}_D{self.head_dim}"


def all_cells():
    """The 47 cells: nine arms at five head sizes, the uniform control, and the unmasked arm at 16."""
    cells = [Cell(arm, D) for arm in CROSSED_ARMS for D in HEAD_SIZES]
    cells.append(Cell("uniform", 16))
    cells.append(Cell("dot_untied_unmasked", 16))
    return cells


# A rough relative cost of one run, used only to order the runs and to estimate the time left.
_ARM_COST = {"dot_tied": 1.0, "dot_untied": 1.1, "dot_untied_unmasked": 1.1, "uniform": 0.9,
             "lp_1": 5.0, "lp_1.5": 10.0, "lp_2": 6.0, "lp_3": 7.0, "lp_4": 5.0, "lp_8": 5.0,
             "lp_inf": 5.0}


def estimated_cost(cell):
    w = _ARM_COST[cell.arm]
    return w if cell.arm.startswith(("dot", "uniform")) else w * (0.5 + cell.head_dim / 32)


# ---------------------------------------------------------------- run folders and jobs

def scan_dir(root, cell, lr):
    return Path(root) / "scan" / f"{cell.id}_lr{lr:g}"


def main_dir(root, cell, seed):
    return Path(root) / "main" / f"{cell.id}_seed{seed}"


@dataclass(frozen=True)
class Job:
    spec: RunSpec
    run_dir: Path
    calib_root: Path
    cost: float
    ckpt_seconds: float


def _spec(cell, seed, lr, stage, st):
    return RunSpec(arm=cell.arm, head_dim=cell.head_dim, seed=seed, lr=lr, stage=stage, steps=st.steps,
                   warmup=st.warmup, eval_every=st.eval_every, log_every=st.log_every)


def scan_jobs(root, st, cells, rates=SCAN_RATES):
    return [Job(_spec(c, st.scan_seed, lr, "scan", st), scan_dir(root, c, lr), Path(root),
                estimated_cost(c), st.ckpt_seconds) for c in cells for lr in rates]


def main_jobs(root, st, cells, selected):
    jobs = []
    for c in cells:
        lr = selected["cells"][c.id]["lr"]
        for seed in st.final_seeds:
            jobs.append(Job(_spec(c, seed, lr, "main", st), main_dir(root, c, seed), Path(root),
                            estimated_cost(c), st.ckpt_seconds))
    return jobs


# ---------------------------------------------------------------- running jobs

def _init_worker():
    import torch
    torch.set_num_threads(1)
    signal.signal(signal.SIGINT, signal.SIG_IGN)     # the parent stops the workers


def _work(job):
    messages = []
    progress_file = job.run_dir / "progress.json"
    try:
        job.run_dir.mkdir(parents=True, exist_ok=True)
        result = run_training(job.spec, job.run_dir, job.calib_root, ckpt_seconds=job.ckpt_seconds,
                              log=messages.append, progress=lambda info: atomic_write_json(progress_file, info))
        return job.run_dir.name, result, None, messages
    except Exception:
        text = traceback.format_exc()
        (job.run_dir / "error.txt").write_text(text)
        return job.run_dir.name, None, text, messages
    finally:
        if progress_file.exists():
            progress_file.unlink()


def _fmt_time(seconds):
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def _active_runs(pending):
    """The runs that have reported progress and are not finished: [(job, progress dict)]."""
    active = []
    for job in pending:
        if (job.run_dir / "result.json").exists():
            continue
        try:
            info = json.loads((job.run_dir / "progress.json").read_text())
        except (OSError, ValueError):
            continue
        active.append((job, info))
    return active


def format_heartbeat(label, n_done, n_total, active, elapsed, eta):
    """One status line: runs finished, the runs in progress with their percentage, and the time left."""
    running = ", ".join(f"{job.run_dir.name} {100 * info['step'] / info['steps']:.0f}%" for job, info in active)
    left = f"about {_fmt_time(eta)} left" if eta is not None else "estimating the time left"
    return (f"{label}: {time.strftime('%H:%M')} | {n_done}/{n_total} finished | "
            f"{running or 'starting'} | elapsed {_fmt_time(elapsed)}, {left}")


def run_jobs(jobs, workers, label, log=print, heartbeat_seconds=HEARTBEAT_SECONDS):
    """Run all unfinished jobs, slowest first. Raises SweepError if any run fails.

    While runs are going, a status line is printed about every heartbeat_seconds.
    """
    pending = [j for j in jobs if not (j.run_dir / "result.json").exists()]
    pending.sort(key=lambda j: -j.cost)
    for job in pending:                                  # progress files of an earlier, interrupted session
        (job.run_dir / "progress.json").unlink(missing_ok=True)
    log(f"{label}: {len(jobs) - len(pending)} of {len(jobs)} runs are finished, {len(pending)} to run")
    if not pending:
        return
    total_cost = sum(j.cost for j in pending)
    done, done_cost, failures, start = 0, 0.0, [], time.time()
    cost_of = {j.run_dir.name: j.cost for j in pending}

    def eta_now(active):
        """The time left, from the cost-weighted share of work done, including runs in progress."""
        progressed = done_cost + sum(j.cost * info["step"] / info["steps"] for j, info in active)
        elapsed = time.time() - start
        return elapsed / progressed * (total_cost - progressed) if progressed > 0 else None

    def report(name, result, error, messages):
        nonlocal done, done_cost
        for m in messages:
            if "resumed" in m:
                log(m)
        done += 1
        done_cost += cost_of[name]
        if error:
            failures.append(name)
            log(f"{label}: {name} FAILED, see {name}/error.txt")
            return
        eta = eta_now(_active_runs(pending))
        bpc = result["final_val_bpc"]
        shown = "diverged" if result["diverged"] else f"val bpc {bpc:.4f}"
        left = "" if eta is None else f", about {_fmt_time(eta)} left"
        log(f"{label}: [{done}/{len(pending)}] {name} {shown}; elapsed {_fmt_time(time.time() - start)}{left}")

    if workers <= 1:
        _init_serial()
        for job in pending:
            report(*_work(job))
    else:
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(workers, initializer=_init_worker)
        last_beat = time.time()

        def beat():
            nonlocal last_beat
            last_beat = time.time()
            active = _active_runs(pending)
            log(format_heartbeat(label, done, len(pending), active, last_beat - start, eta_now(active)))

        try:
            results = pool.imap_unordered(_work, pending, chunksize=1)
            while done < len(pending):
                wait = max(0.05, heartbeat_seconds - (time.time() - last_beat))
                try:
                    report(*results.next(timeout=wait))
                except multiprocessing.TimeoutError:
                    pass
                if time.time() - last_beat >= heartbeat_seconds:
                    beat()
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            pool.terminate()
            raise
        except BaseException:
            pool.terminate()
            raise
    if failures:
        raise SweepError(f"{label}: {len(failures)} runs failed: {', '.join(sorted(failures))}")


def _init_serial():
    import torch
    torch.set_num_threads(1)


# ---------------------------------------------------------------- the scan and the selection

def _read_result(path):
    path = Path(path) / "result.json"
    return json.loads(path.read_text()) if path.exists() else None


def scan_table(root, cell):
    """{rate: result or None} over the base grid and the two extra rates."""
    rates = SCAN_RATES + (EXTRA_RATE_LOW, EXTRA_RATE_HIGH)
    return {lr: _read_result(scan_dir(root, cell, lr)) for lr in rates}


def choose_rate(table):
    """Apply the protocol of design section 8 to the results of one cell.

    Returns (chosen rate or None, the extra rate still needed or None, True if the base grid is incomplete).
    """
    base = {lr: table[lr] for lr in SCAN_RATES}
    if any(r is None for r in base.values()):
        return None, None, True

    def ok(rates):
        return {lr: table[lr]["final_val_bpc"] for lr in rates
                if table[lr] is not None and not table[lr]["diverged"]}

    def best(candidates):
        return min(candidates, key=lambda lr: (candidates[lr], lr)) if candidates else None

    base_best = best(ok(SCAN_RATES))
    extra = None
    if base_best == min(SCAN_RATES):
        extra = EXTRA_RATE_LOW
    elif base_best == max(SCAN_RATES):
        extra = EXTRA_RATE_HIGH
    if extra is not None and table[extra] is not None:
        extra = None                                   # already run
    needed = extra
    chosen = best(ok(tuple(table)))
    return chosen, needed, False


def select_rates(root, cells, log=print):
    """Choose the rate of every cell from the scan, and write selected_rates.json and scan_table.csv."""
    selected = {"cells": {}, "needs_extra": [], "incomplete": [], "all_diverged": []}
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["cell", "lr", "val_bpc", "diverged", "chosen"])
    for cell in cells:
        table = scan_table(root, cell)
        chosen, needed, incomplete = choose_rate(table)
        if incomplete:
            selected["incomplete"].append(cell.id)
            continue
        if needed is not None:
            selected["needs_extra"].append({"cell": cell.id, "lr": needed})
        if chosen is None:
            selected["all_diverged"].append(cell.id)
        selected["cells"][cell.id] = {
            "arm": cell.arm, "D": cell.head_dim, "lr": chosen,
            "val_bpc": table[chosen]["final_val_bpc"] if chosen is not None else None,
            "scan": {f"{lr:g}": {"val_bpc": r["final_val_bpc"], "diverged": r["diverged"]}
                     for lr, r in table.items() if r is not None},
        }
        for lr, r in table.items():
            if r is not None:
                writer.writerow([cell.id, f"{lr:g}", "" if r["final_val_bpc"] is None else r["final_val_bpc"],
                                 r["diverged"], lr == chosen])
    selected["complete"] = not (selected["incomplete"] or selected["needs_extra"] or selected["all_diverged"])
    atomic_write_json(Path(root) / "selected_rates.json", selected)
    atomic_write_text(Path(root) / "scan_table.csv", out.getvalue())
    log(f"select: {len(selected['cells'])} cells chosen; {len(selected['incomplete'])} incomplete, "
        f"{len(selected['needs_extra'])} need an extra rate, {len(selected['all_diverged'])} diverged at every rate")
    for item in selected["needs_extra"]:
        log(f"select: {item['cell']} needs the extra rate {item['lr']:g}")
    for cid in selected["all_diverged"]:
        log(f"select: {cid} diverged at every rate and has no rate")
    return selected


def run_scan(root, st, cells, log=print):
    """Stage 1: the scan at three rates, then the one extra rate for each cell whose best rate is at an end."""
    run_jobs(scan_jobs(root, st, cells), st.workers, "scan", log)
    needed = select_rates(root, cells, log=lambda m: None)["needs_extra"]
    by_id = {c.id: c for c in cells}
    extras = [Job(_spec(by_id[n["cell"]], st.scan_seed, n["lr"], "scan", st), scan_dir(root, by_id[n["cell"]], n["lr"]),
                  Path(root), estimated_cost(by_id[n["cell"]]), st.ckpt_seconds) for n in needed]
    if extras:
        run_jobs(extras, st.workers, "scan (extra rates)", log)


def run_select(root, cells, log=print):
    selected = select_rates(root, cells, log=log)
    if not selected["complete"]:
        raise SweepError("the selection is not complete; run the scan stage first")
    return selected


def run_main(root, st, cells, log=print):
    path = Path(root) / "selected_rates.json"
    if not path.exists():
        raise SweepError("selected_rates.json is missing. Run the select stage first.")
    selected = json.loads(path.read_text())
    if not selected["complete"]:
        raise SweepError("selected_rates.json is not complete. Run the scan and select stages first.")
    run_jobs(main_jobs(root, st, cells, selected), st.workers, "main", log)


# ---------------------------------------------------------------- the smoke run

SMOKE_CELLS = (Cell("lp_8", 2), Cell("dot_tied", 4))
SMOKE_SETTINGS = Settings(steps=20, warmup=4, eval_every=10, log_every=5, final_seeds=(1, 2), workers=2,
                          ckpt_seconds=600, calibration_batches=2)


def run_smoke(root, log=print):
    """Every stage on a tiny scale: two cells, 20 steps, results in a separate folder."""
    root = Path(root) / "smoke"
    st, cells = SMOKE_SETTINGS, list(SMOKE_CELLS)
    log(f"smoke: results go to {root}")
    run_calibration(root, n_batches=st.calibration_batches, combos=[(c.arm, c.head_dim) for c in cells], log=log)
    run_scan(root, st, cells, log)
    run_select(root, cells, log)
    run_main(root, st, cells, log)
    # The two smoke cells are not a full grid, so the analysis is smoke-tested on invented results.
    from .analyze import run_analysis, write_invented_results
    write_invented_results(root / "invented")
    run_analysis(root / "invented", log=log)
    return root, st, cells
