"""Run many training runs, optionally in parallel worker processes. Completed runs are skipped and
interrupted runs resume from their checkpoint, so re-issuing the same command continues the job."""
import concurrent.futures as cf
import multiprocessing as mp
import time
import traceback


def _work(args):
    cfg, out_root, tag = args
    from . import train
    try:
        return train.run(cfg, out_root, tag=tag, verbose=False)
    except Exception:  # noqa
        return {"run_id": train.run_id(cfg, tag), "error": traceback.format_exc()}


def run_many(cfgs, out_root, workers=1, tag="", progress=True):
    """cfgs: list of config dicts (each with its own 'threads'). Returns results in the same order."""
    from . import train
    t0 = time.time()
    jobs = [(c, str(out_root), tag) for c in cfgs]
    results = [None] * len(jobs)
    if workers <= 1:
        for i, j in enumerate(jobs):
            results[i] = _work(j)
            if progress:
                print(f"  [{i + 1}/{len(jobs)}] {train.run_id(j[0], tag)} done ({time.time() - t0:.0f}s)", flush=True)
        return results
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = {ex.submit(_work, j): i for i, j in enumerate(jobs)}
        done = 0
        for f in cf.as_completed(futs):
            i = futs[f]
            results[i] = f.result()
            done += 1
            if progress:
                print(f"  [{done}/{len(jobs)}] {train.run_id(jobs[i][0], tag)} done ({time.time() - t0:.0f}s)", flush=True)
    return results
