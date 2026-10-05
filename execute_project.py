#!/usr/bin/env python3
"""The single entry point of the project. Run `python execute_project.py <stage>`.

Stages
  check      run the unit tests
  smoke      every stage on a tiny scale (two cells, 20 steps); results in results/smoke/
  calibrate  measure the score constants
  scan       learning-rate scan (3 rates for each of the 47 cells, then one extra rate where needed)
  select     choose the rate of each cell from the scan
  main       five seeds for each cell at its chosen rate
  analyze    tables, tests and figures from the results
  all        calibrate, scan, select, main and analyze, in this order

The script never runs git. Every stage can be interrupted with Ctrl-C and started again: finished
runs are skipped, and an unfinished run continues from its last checkpoint (written every 10 minutes).
On a laptop, keep the machine awake during a long run, for example with `caffeinate -i python execute_project.py all`.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = ["check", "smoke", "calibrate", "scan", "select", "main", "analyze", "all"]
ALL_ORDER = ["calibrate", "scan", "select", "main", "analyze"]


def stage_check(args):
    return subprocess.call([sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q"], cwd=ROOT)


def stage_smoke(args):
    from msa.sweep import run_smoke
    run_smoke(args.results)
    return 0


def stage_calibrate(args):
    from msa.calibrate import run_calibration
    run_calibration(args.results)
    return 0


def stage_scan(args):
    from msa.sweep import Settings, all_cells, run_scan
    run_scan(args.results, Settings(workers=args.workers), all_cells())
    return 0


def stage_select(args):
    from msa.sweep import all_cells, run_select
    run_select(args.results, all_cells())
    return 0


def stage_main(args):
    from msa.sweep import Settings, all_cells, run_main
    run_main(args.results, Settings(workers=args.workers), all_cells())
    return 0


def stage_analyze(args):
    from msa.analyze import run_analysis
    run_analysis(args.results)
    return 0


def stage_all(args):
    for name in ALL_ORDER:
        print(f"=== {name} ===", flush=True)
        code = STAGE_FUNCTIONS[name](args)
        if code:
            return code
    return 0


STAGE_FUNCTIONS = {"check": stage_check, "smoke": stage_smoke, "calibrate": stage_calibrate,
                   "scan": stage_scan, "select": stage_select, "main": stage_main,
                   "analyze": stage_analyze, "all": stage_all}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--results", type=Path, default=ROOT / "results",
                        help="folder for all results (default: results/ in the repo)")
    parser.add_argument("--workers", type=int, default=4, help="runs in parallel, one thread each (default 4)")
    args = parser.parse_args(argv)
    try:
        return STAGE_FUNCTIONS[args.stage](args)
    except KeyboardInterrupt:
        print("\nInterrupted. Run the same command again to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
