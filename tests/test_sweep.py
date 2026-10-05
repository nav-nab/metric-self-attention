import json

import pytest

from msa.calibrate import run_calibration
from msa.sweep import (EXTRA_RATE_HIGH, EXTRA_RATE_LOW, SCAN_RATES, Cell, Settings, SweepError, all_cells,
                       choose_rate, estimated_cost, main_jobs, run_jobs, run_main, run_scan, run_select,
                       scan_dir, scan_jobs, select_rates)

LOW, MID, HIGH = SCAN_RATES


def res(bpc, diverged=False):
    return {"final_val_bpc": bpc, "diverged": diverged}


def table(low=None, mid=None, high=None, xlow=None, xhigh=None):
    return {LOW: low, MID: mid, HIGH: high, EXTRA_RATE_LOW: xlow, EXTRA_RATE_HIGH: xhigh}


def test_there_are_47_cells_141_scan_runs_and_235_main_runs():
    cells = all_cells()
    assert len(cells) == 47 and len({c.id for c in cells}) == 47
    assert Cell("dot_untied_unmasked", 16) in cells and Cell("uniform", 16) in cells
    assert len(scan_jobs("r", Settings(), cells)) == 141
    selected = {"cells": {c.id: {"lr": 1e-2} for c in cells}}
    assert len(main_jobs("r", Settings(), cells, selected)) == 235


def test_jobs_use_the_right_seeds_and_the_slowest_come_first():
    cells = all_cells()
    scan = scan_jobs("r", Settings(), cells)
    assert {j.spec.seed for j in scan} == {100} and {j.spec.stage for j in scan} == {"scan"}
    selected = {"cells": {c.id: {"lr": 1e-2} for c in cells}}
    main = main_jobs("r", Settings(), cells, selected)
    assert {j.spec.seed for j in main} == {1, 2, 3, 4, 5}
    assert estimated_cost(Cell("lp_1.5", 16)) > estimated_cost(Cell("lp_1.5", 2)) > estimated_cost(Cell("dot_tied", 16))


def test_rate_choice_follows_design_section_8():
    # best rate inside the grid: chosen, no extra rate
    assert choose_rate(table(res(2.5), res(2.4), res(2.6))) == (MID, None, False)
    # best at the low end: the extra rate 1e-3 is needed
    assert choose_rate(table(res(2.4), res(2.5), res(2.6))) == (LOW, EXTRA_RATE_LOW, False)
    # best at the high end: the extra rate 9e-2 is needed
    assert choose_rate(table(res(2.6), res(2.5), res(2.4))) == (HIGH, EXTRA_RATE_HIGH, False)
    # the extra rate was run and is better: it is chosen, and nothing more is needed
    assert choose_rate(table(res(2.6), res(2.5), res(2.4), xhigh=res(2.3))) == (EXTRA_RATE_HIGH, None, False)
    # the extra rate was run and is worse (or diverged): the end of the grid stays chosen
    assert choose_rate(table(res(2.6), res(2.5), res(2.4), xhigh=res(2.45))) == (HIGH, None, False)
    assert choose_rate(table(res(2.6), res(2.5), res(2.4), xhigh=res(9.0, True))) == (HIGH, None, False)
    # a diverged rate is never chosen: the best of the others is at the end of the grid
    assert choose_rate(table(res(2.6), res(2.5), res(None, True))) == (MID, None, False)
    assert choose_rate(table(res(2.4), res(None, True), res(None, True))) == (LOW, EXTRA_RATE_LOW, False)
    # all rates diverged: no rate and no extra
    assert choose_rate(table(res(None, True), res(None, True), res(None, True))) == (None, None, False)
    # an incomplete base grid
    assert choose_rate(table(res(2.4), None, res(2.6)))[2] is True


def write_scan_result(root, cell, lr, bpc, diverged=False):
    d = scan_dir(root, cell, lr)
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"final_val_bpc": bpc, "diverged": diverged}))


def test_selection_files(tmp_path):
    a, b, c = Cell("dot_tied", 2), Cell("lp_2", 2), Cell("lp_1", 2)
    for lr, bpc in zip(SCAN_RATES, (2.5, 2.4, 2.6)):
        write_scan_result(tmp_path, a, lr, bpc)                     # interior best
    for lr, bpc in zip(SCAN_RATES, (2.4, 2.5, 2.6)):
        write_scan_result(tmp_path, b, lr, bpc)                     # needs the low extra rate
    write_scan_result(tmp_path, c, LOW, 2.4)                        # incomplete
    sel = select_rates(tmp_path, [a, b, c], log=lambda m: None)
    assert sel["cells"][a.id]["lr"] == MID and sel["needs_extra"] == [{"cell": b.id, "lr": EXTRA_RATE_LOW}]
    assert sel["incomplete"] == [c.id] and sel["complete"] is False
    assert (tmp_path / "scan_table.csv").read_text().count("\n") == 1 + 6
    with pytest.raises(SweepError):
        run_main(tmp_path, Settings(), [a], log=lambda m: None)
    with pytest.raises(SweepError):
        run_select(tmp_path, [a, b, c], log=lambda m: None)


TINY = Settings(steps=6, warmup=2, eval_every=3, log_every=3, final_seeds=(1, 2), workers=2, ckpt_seconds=600,
                calibration_batches=2)


def test_the_stages_run_end_to_end_resume_and_report_failures(tmp_path):
    cells = [Cell("dot_tied", 2), Cell("lp_1.5", 4), Cell("uniform", 16)]
    run_calibration(tmp_path, n_batches=2, combos=[("dot_tied", 2), ("lp_1.5", 4)], log=lambda m: None)
    msgs = []
    run_scan(tmp_path, TINY, cells, msgs.append)
    sel = run_select(tmp_path, cells, log=lambda m: None)
    scan_runs = sorted(p.name for p in (tmp_path / "scan").iterdir())
    assert len(scan_runs) >= 9 and all((tmp_path / "scan" / n / "result.json").exists() for n in scan_runs)
    assert sel["complete"] and all(v["lr"] is not None for v in sel["cells"].values())
    run_main(tmp_path, TINY, cells, msgs.append)
    main_runs = sorted(p.name for p in (tmp_path / "main").iterdir())
    assert len(main_runs) == 6
    chosen = sel["cells"]["lp_1.5_D4"]["lr"]
    assert json.loads((tmp_path / "main" / "lp_1.5_D4_seed1" / "result.json").read_text())["lr"] == chosen
    assert not list(tmp_path.rglob("ckpt.pt")) and not list(tmp_path.rglob("*.tmp"))

    # a second call starts nothing
    stamps = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("result.json")}
    msgs.clear()
    run_scan(tmp_path, TINY, cells, msgs.append)
    run_main(tmp_path, TINY, cells, msgs.append)
    assert {p: p.stat().st_mtime_ns for p in tmp_path.rglob("result.json")} == stamps
    assert all("0 to run" in m for m in msgs)

    # a removed result is run again, and only that one
    victim = tmp_path / "main" / "dot_tied_D2_seed2"
    (victim / "result.json").unlink()
    run_main(tmp_path, TINY, cells, msgs.append)
    assert (victim / "result.json").exists()
    assert {p: p.stat().st_mtime_ns for p in tmp_path.rglob("result.json") if victim not in p.parents} == \
           {p: s for p, s in stamps.items() if victim not in p.parents}


def test_a_failing_run_is_reported_and_does_not_stop_the_others(tmp_path):
    cells = [Cell("dot_tied", 2), Cell("lp_2", 4)]          # no calibration file for lp_2
    run_calibration(tmp_path, n_batches=2, combos=[("dot_tied", 2)], log=lambda m: None)
    st = Settings(**{**TINY.__dict__, "workers": 1})
    with pytest.raises(SweepError) as err:
        run_jobs(scan_jobs(tmp_path, st, cells, rates=(1e-2,)), 1, "scan", log=lambda m: None)
    assert "lp_2_D4_lr0.01" in str(err.value)
    assert (scan_dir(tmp_path, cells[0], 1e-2) / "result.json").exists()
    assert "calibrate stage" in (scan_dir(tmp_path, cells[1], 1e-2) / "error.txt").read_text()


def test_heartbeat_line_and_progress_files(tmp_path):
    from msa.sweep import _active_runs, format_heartbeat
    cells = [Cell("dot_tied", 2)]
    jobs = scan_jobs(tmp_path, TINY, cells, rates=(1e-2, 3e-2))
    d = jobs[0].run_dir
    d.mkdir(parents=True)
    (d / "progress.json").write_text(json.dumps({"step": 1000, "steps": 4000, "train_loss": 2.5, "val_bpc": None}))
    (jobs[1].run_dir).mkdir(parents=True)
    (jobs[1].run_dir / "progress.json").write_text("{broken")           # a half-written file is ignored
    active = _active_runs(jobs)
    assert [j.run_dir.name for j, _ in active] == ["dot_tied_D2_lr0.01"]
    line = format_heartbeat("scan", 3, 141, active, 3725, 7200)
    assert "3/141 finished" in line and "dot_tied_D2_lr0.01 25%" in line
    assert "elapsed 1h02m" in line and "about 2h00m left" in line
    assert "estimating" in format_heartbeat("scan", 0, 5, [], 10, None)


def test_a_status_line_is_printed_while_runs_are_going(tmp_path):
    cells = [Cell("dot_tied", 2), Cell("lp_2", 4)]
    run_calibration(tmp_path, n_batches=2, combos=[("dot_tied", 2), ("lp_2", 4)], log=lambda m: None)
    st = Settings(steps=60, warmup=4, eval_every=30, log_every=10, workers=2, ckpt_seconds=600, calibration_batches=2)
    msgs = []
    run_jobs(scan_jobs(tmp_path, st, cells, rates=(1e-2,)), 2, "scan", msgs.append, heartbeat_seconds=0.3)
    beats = [m for m in msgs if "finished |" in m]
    assert beats and any("%" in m for m in beats)
    assert not list(tmp_path.rglob("progress.json"))                     # removed when the runs end
