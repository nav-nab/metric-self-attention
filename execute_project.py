#!/usr/bin/env python3
"""The single entry point of the project. Run `python execute_project.py <stage>`.

Stages: check, smoke, calibrate, scan, select, main, analyze, all.
The script never runs git. Every stage can be interrupted and started again.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = ["check", "smoke", "calibrate", "scan", "select", "main", "analyze", "all"]
BUILT = {"check", "calibrate"}


def stage_check(args):
    """Run the unit tests."""
    return subprocess.call([sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q"], cwd=ROOT)


def stage_calibrate(args):
    from msa.calibrate import run_calibration
    run_calibration(args.results)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--results", type=Path, default=ROOT / "results",
                        help="folder for all results (default: results/ in the repo)")
    args = parser.parse_args(argv)
    if args.stage not in BUILT:
        print(f"stage '{args.stage}' is not written yet", file=sys.stderr)
        return 2
    return {"check": stage_check, "calibrate": stage_calibrate}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
