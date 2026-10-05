import pytest

import execute_project
from msa import sweep


def test_all_runs_the_stages_in_order_and_stops_at_a_failure(monkeypatch):
    calls = []
    fake = {name: (lambda a, n=name: calls.append(n) or 0) for name in execute_project.ALL_ORDER}
    monkeypatch.setattr(execute_project, "STAGE_FUNCTIONS", {**execute_project.STAGE_FUNCTIONS, **fake})
    assert execute_project.main(["all"]) == 0
    assert calls == ["calibrate", "scan", "select", "main", "analyze"]
    calls.clear()
    fake["select"] = lambda a: calls.append("select") or 1
    monkeypatch.setattr(execute_project, "STAGE_FUNCTIONS", {**execute_project.STAGE_FUNCTIONS, **fake})
    assert execute_project.main(["all"]) == 1 and calls == ["calibrate", "scan", "select"]


def test_ctrl_c_prints_how_to_resume(monkeypatch, capsys):
    def interrupted(args):
        raise KeyboardInterrupt
    monkeypatch.setitem(execute_project.STAGE_FUNCTIONS, "scan", interrupted)
    assert execute_project.main(["scan"]) == 130
    assert "Run the same command again to resume" in capsys.readouterr().err


def test_the_stage_list_matches_the_plan():
    assert execute_project.STAGES == ["check", "smoke", "calibrate", "scan", "select", "main", "analyze", "all"]
    assert set(execute_project.STAGES) == set(execute_project.STAGE_FUNCTIONS)


def test_smoke_uses_a_separate_folder_two_cells_and_tiny_settings(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sweep, "run_calibration", lambda root, **kw: calls.append(("calibrate", root, kw["combos"])))
    monkeypatch.setattr(sweep, "run_scan", lambda root, st, cells, log: calls.append(("scan", root)))
    monkeypatch.setattr(sweep, "run_select", lambda root, cells, log: calls.append(("select", root)))
    monkeypatch.setattr(sweep, "run_main", lambda root, st, cells, log: calls.append(("main", root)))
    from msa import analyze
    monkeypatch.setattr(analyze, "write_invented_results", lambda root: calls.append(("invented", root)))
    monkeypatch.setattr(analyze, "run_analysis", lambda root, log: calls.append(("analyze", root)))
    root, st, cells = sweep.run_smoke(tmp_path, log=lambda m: None)
    assert root == tmp_path / "smoke" and len(cells) == 2 and st.steps == 20
    assert [c[0] for c in calls] == ["calibrate", "scan", "select", "main", "invented", "analyze"]
    assert all(c[1] == tmp_path / "smoke" for c in calls[:4])
    assert all(c[1] == tmp_path / "smoke" / "invented" for c in calls[4:])
