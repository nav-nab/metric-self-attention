import json
import math

import pytest
import torch

from msa.calibrate import run_calibration
from msa.data import load_corpus
from msa.model import build_model
from msa.train import (RunInterrupted, RunSpec, is_diverged, lr_at, make_optimizer, run_training)

SMALL = dict(steps=24, warmup=4, eval_every=8, log_every=4)


@pytest.fixture(scope="module")
def calib(tmp_path_factory):
    root = tmp_path_factory.mktemp("calib")
    run_calibration(root, n_batches=2, combos=[("lp_2", 4), ("dot_tied", 4), ("lp_8", 2)], log=lambda m: None)
    return root


def spec_for(arm="lp_2", D=4, lr=3e-3, seed=1, **kw):
    return RunSpec(arm=arm, head_dim=D, seed=seed, lr=lr, stage="smoke", **{**SMALL, **kw})


def test_learning_rate_schedule():
    s = RunSpec("lp_2", 4, 1, 1e-2, "main")
    assert lr_at(s, 0) == pytest.approx(1e-2 / 200)
    assert lr_at(s, 199) == pytest.approx(1e-2)
    assert lr_at(s, 3999) == pytest.approx(1e-3)
    later = [lr_at(s, t) for t in range(200, 4000)]
    assert all(a >= b for a, b in zip(later, later[1:]))


def test_weight_decay_applies_to_linear_weight_matrices_only():
    model = build_model("dot_untied", 4, 1)
    opt = make_optimizer(model, 1e-3)
    decayed = {id(p) for g in opt.param_groups if g["weight_decay"] > 0 for p in g["params"]}
    names = {id(p): n for n, p in model.named_parameters()}
    got = sorted(names[i] for i in decayed)
    assert got and all(n.endswith(".weight") and any(k in n for k in ("w_q", "w_k", "w_v", "w_o", "fc1", "fc2"))
                       for n in got)
    assert len(got) == 2 * 6          # six matrices per layer, two layers
    for n, p in model.named_parameters():
        if "tok_emb" in n or "pos_emb" in n or "ln" in n or "log_beta" in n or n.endswith("bias"):
            assert id(p) not in decayed


def test_divergence_rule():
    assert not is_diverged(True, 2.5, 65)
    assert is_diverged(True, math.log2(65) + 0.01, 65)
    assert is_diverged(False, None, 65)


def test_a_run_writes_its_files_and_is_then_skipped(tmp_path, calib):
    spec = spec_for()
    msgs = []
    result = run_training(spec, tmp_path / "r", calib, log=msgs.append)
    files = {p.name for p in (tmp_path / "r").iterdir()}
    assert files == {"result.json", "log.csv", "final.pt", "diagnostics.json"}   # no ckpt, no tmp
    assert result["diverged"] is False and result["steps_done"] == 24 and result["resumes"] == 0
    assert 3.0 < result["final_val_bpc"] < 6.1
    diag = json.loads((tmp_path / "r" / "diagnostics.json").read_text())
    assert [d["step"] for d in diag] == [8, 16, 24]
    layer = diag[-1]["layers"][0]
    assert len(layer["beta"]) == 4 and all(v > 0 for v in layer["entropy"])
    stamp = (tmp_path / "r" / "result.json").stat().st_mtime_ns
    again = run_training(spec, tmp_path / "r", calib, log=msgs.append)
    assert again == result and (tmp_path / "r" / "result.json").stat().st_mtime_ns == stamp


@pytest.mark.parametrize("arm,D", [("lp_2", 4), ("dot_tied", 4), ("lp_8", 2)])
def test_interrupted_run_resumes_and_agrees_with_the_uninterrupted_run(tmp_path, calib, arm, D):
    spec = spec_for(arm, D)
    ref = run_training(spec, tmp_path / "ref", calib, log=lambda m: None)

    # A fake clock that reads the number of finished steps. With a 5-second interval, the
    # checkpoints are written after steps 5, 10, 15 and 20. The interruption comes after step 13.
    now = {"t": 0}

    def hook(done):
        now["t"] = done
        if done == 13:
            raise RunInterrupted

    run_dir = tmp_path / "cut"
    with pytest.raises(RunInterrupted):
        run_training(spec, run_dir, calib, ckpt_seconds=5, clock=lambda: now["t"], stop_hook=hook,
                     log=lambda m: None)
    assert not (run_dir / "result.json").exists()
    assert torch.load(run_dir / "ckpt.pt")["step"] == 10        # work after step 10 is lost
    assert not list(run_dir.glob("*.tmp"))

    msgs = []
    now["t"] = 0
    res = run_training(spec, run_dir, calib, ckpt_seconds=5, clock=lambda: now["t"], log=msgs.append)
    assert any("resumed from step 10" in m for m in msgs)
    assert res["resumes"] == 1 and res["steps_done"] == 24
    assert res["final_val_bpc"] == pytest.approx(ref["final_val_bpc"], abs=1e-6)
    assert res["final_train_loss"] == pytest.approx(ref["final_train_loss"], abs=1e-6)
    a = torch.load(tmp_path / "ref" / "final.pt")["model"]
    b = torch.load(run_dir / "final.pt")["model"]
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=1e-5, atol=1e-7)
    assert (tmp_path / "ref" / "log.csv").read_text() == (run_dir / "log.csv").read_text()
    assert not (run_dir / "ckpt.pt").exists()


def test_a_checkpoint_of_another_run_is_refused(tmp_path, calib):
    now = {"t": 0}

    def hook(done):
        now["t"] = done
        if done == 6:
            raise RunInterrupted

    with pytest.raises(RunInterrupted):
        run_training(spec_for(), tmp_path / "r", calib, ckpt_seconds=5, clock=lambda: now["t"], stop_hook=hook,
                     log=lambda m: None)
    with pytest.raises(RuntimeError):
        run_training(spec_for(lr=1e-2), tmp_path / "r", calib, log=lambda m: None)


def test_a_non_finite_loss_is_recorded_as_divergence(tmp_path, calib):
    result = run_training(spec_for(lr=1e30), tmp_path / "d", calib, log=lambda m: None)
    assert result["diverged"] and result["final_val_bpc"] is None
    assert result["diverged_reason"] == "non-finite training loss"
    assert not (tmp_path / "d" / "final.pt").exists() and not (tmp_path / "d" / "ckpt.pt").exists()


def test_the_uniform_arm_trains_without_calibration(tmp_path):
    result = run_training(spec_for("uniform", 16), tmp_path / "u", tmp_path, log=lambda m: None)
    assert result["D"] is None and not result["diverged"]


def test_a_killed_process_resumes_from_its_last_checkpoint(tmp_path, calib):
    """A real process is killed with SIGKILL while it writes checkpoints, then the run is resumed."""
    import subprocess
    import sys
    import time
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec_args = "arm='lp_2', head_dim=4, seed=1, lr=3e-3, stage='smoke', steps=80, warmup=4, eval_every=40, log_every=4"
    code = (f"import sys; sys.path.insert(0, {str(root)!r}); from msa.train import RunSpec, run_training; "
            f"run_training(RunSpec({spec_args}), {str(tmp_path / 'killed')!r}, {str(calib)!r}, "
            f"ckpt_seconds=0, log=lambda m: None)")
    proc = subprocess.Popen([sys.executable, "-c", code])
    ckpt = tmp_path / "killed" / "ckpt.pt"
    deadline = time.time() + 120
    while time.time() < deadline and proc.poll() is None:
        try:
            if ckpt.exists() and torch.load(ckpt)["step"] >= 12:
                break
        except Exception:
            pass                                  # the file was being replaced; look again
        time.sleep(0.05)
    assert proc.poll() is None, "the run finished before it could be killed"
    proc.kill()
    proc.wait()
    assert not (tmp_path / "killed" / "result.json").exists()
    saved = torch.load(ckpt)["step"]              # an intact checkpoint survived the kill
    assert 12 <= saved < 80

    spec = RunSpec(arm="lp_2", head_dim=4, seed=1, lr=3e-3, stage="smoke", steps=80, warmup=4,
                   eval_every=40, log_every=4)
    msgs = []
    res = run_training(spec, tmp_path / "killed", calib, log=msgs.append)
    ref = run_training(spec, tmp_path / "ref", calib, log=lambda m: None)
    assert any("resumed from step" in m for m in msgs) and res["resumes"] == 1
    assert res["final_val_bpc"] == pytest.approx(ref["final_val_bpc"], abs=1e-6)
    assert not list((tmp_path / "killed").glob("*.tmp"))


def test_progress_callback_reports_every_logged_step(tmp_path, calib):
    seen = []
    run_training(spec_for(), tmp_path / "p", calib, log=lambda m: None, progress=seen.append)
    assert [s["step"] for s in seen] == [4, 8, 12, 16, 20, 24]
    assert all(s["steps"] == 24 and math.isfinite(s["train_loss"]) for s in seen)
    assert seen[0]["val_bpc"] is None and seen[1]["val_bpc"] is not None       # evaluation at step 8
