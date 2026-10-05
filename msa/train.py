"""One training run, with evaluation, diagnostics and resumable checkpoints.

A run is finished when result.json exists. A running run writes a checkpoint at most
every CKPT_SECONDS of wall-clock time, atomically, and a restarted run continues from it.
The data stream is a pure function of (seed, step), so a resumed run sees the same batches.
"""
import csv
import io
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .calibrate import MIN_KEYS, load_calibration
from .data import BATCH, load_corpus, train_batch, validation_windows
from .model import build_model
from .scores import parse_arm
from .util import atomic_torch_save, atomic_write_json, atomic_write_text

STEPS = 4000
WARMUP = 200
MIN_LR_FRACTION = 0.1
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.01
CLIP_NORM = 1.0
EVAL_EVERY = 500
LOG_EVERY = 50
CKPT_SECONDS = 600           # the 10 minutes of wall-clock time of plan section 4
FINAL_LOSS_WINDOW = 100      # final training loss: mean over the last steps
DIAG_WINDOWS = 64            # validation windows used for the attention diagnostics


@dataclass(frozen=True)
class RunSpec:
    arm: str
    head_dim: int
    seed: int
    lr: float
    stage: str                 # "scan", "main" or "smoke"
    steps: int = STEPS
    warmup: int = WARMUP
    eval_every: int = EVAL_EVERY
    log_every: int = LOG_EVERY


class RunInterrupted(Exception):
    """Raised by a test hook to stop a run in the middle."""


def lr_at(spec, step):
    """Linear warm-up, then cosine decay to MIN_LR_FRACTION of the peak rate."""
    if step < spec.warmup:
        return spec.lr * (step + 1) / spec.warmup
    progress = (step - spec.warmup) / max(1, spec.steps - spec.warmup - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return spec.lr * (MIN_LR_FRACTION + (1.0 - MIN_LR_FRACTION) * cosine)


def is_diverged(loss_is_finite, final_val_bpc, vocab_size):
    """Plan section 5: a non-finite training loss, or a final bpc above that of a uniform guess."""
    return (not loss_is_finite) or final_val_bpc is None or final_val_bpc > math.log2(vocab_size)


def make_optimizer(model, lr):
    """AdamW. Weight decay applies to the weight matrices of the linear layers only."""
    decay, no_decay = [], []
    linear_weights = {id(m.weight) for m in model.modules() if isinstance(m, nn.Linear)}
    for p in model.parameters():
        (decay if id(p) in linear_weights else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": WEIGHT_DECAY},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=BETAS)


@torch.no_grad()
def evaluate(model, windows, batch=512):
    """Validation bits per character over all given windows."""
    model.eval()
    x, y = windows
    total, count = 0.0, 0
    for i in range(0, len(x), batch):
        _, loss = model(x[i:i + batch], y[i:i + batch])
        n = y[i:i + batch].numel()
        total += loss.item() * n
        count += n
    return total / count / math.log(2)


@torch.no_grad()
def diagnostics(model, windows):
    """Per layer and head: the scale beta, the largest logit, the attention entropy, q and k norms."""
    if not parse_arm(model.cfg.arm).has_scores:
        return None
    model.eval()
    x, _ = windows
    for block in model.blocks:
        block.attn.capture = True
    model(x[:DIAG_WINDOWS])
    out = []
    for block in model.blocks:
        attn = block.attn
        cap = attn.captured
        T = cap["attn"].shape[-1]
        rows = attn.allowed[:T, :T].sum(-1) >= MIN_KEYS
        a = cap["attn"][:, :, rows]                       # (B, H, R, T)
        entropy = -(a * torch.where(a > 0, a.log(), torch.zeros_like(a))).sum(-1).mean(dim=(0, 2))
        out.append({
            "beta": attn.log_beta.exp().tolist(),
            "max_logit": cap["scores"].amax(dim=(0, 2, 3)).tolist(),
            "entropy": entropy.tolist(),
            "q_norm": cap["q"].norm(dim=-1).mean(dim=(0, 2)).tolist(),
            "k_norm": cap["k"].norm(dim=-1).mean(dim=(0, 2)).tolist(),
        })
        attn.capture = False
        attn.captured = None
    return out


def _log_csv(rows):
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["step", "train_loss", "lr", "val_bpc"])
    for r in rows:
        w.writerow([r["step"], r.get("train_loss", ""), r.get("lr", ""), r.get("val_bpc", "")])
    return out.getvalue()


def run_training(spec, run_dir, calib_root, ckpt_seconds=CKPT_SECONDS, clock=time.monotonic,
                 stop_hook=None, log=print):
    """Train one model and write its files. A finished run is skipped. Returns result.json as a dict."""
    run_dir = Path(run_dir)
    result_path = run_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in run_dir.glob("*.tmp"):
        stale.unlink()

    corpus = load_corpus()
    windows = validation_windows(corpus)
    calibration = None
    if parse_arm(spec.arm).has_scores:
        calibration = load_calibration(calib_root, spec.arm, spec.head_dim)
    model = build_model(spec.arm, spec.head_dim, spec.seed, corpus.vocab_size, calibration)
    opt = make_optimizer(model, spec.lr)

    start, train_losses, rows, diag, elapsed, resumes = 0, [], [], [], 0.0, 0
    ckpt_path = run_dir / "ckpt.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path)
        if ckpt["spec"] != asdict(spec):
            raise RuntimeError(f"{ckpt_path} belongs to a different run: {ckpt['spec']}")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        torch.set_rng_state(ckpt["rng"])
        start, train_losses, rows = ckpt["step"], ckpt["train_losses"], ckpt["rows"]
        diag, elapsed, resumes = ckpt["diag"], ckpt["elapsed"], ckpt["resumes"] + 1
        log(f"{run_dir.name}: resumed from step {start}")

    t0 = clock()
    last_ckpt = t0
    params = [p for p in model.parameters()]
    final_val, loss_ok = None, True
    for step in range(start, spec.steps):
        x, y = train_batch(corpus, spec.seed, step)
        lr = lr_at(spec, step)
        for g in opt.param_groups:
            g["lr"] = lr
        model.train()
        _, loss = model(x, y)
        value = loss.item()
        if not math.isfinite(value):
            loss_ok = False
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, CLIP_NORM)
        opt.step()
        train_losses.append(value)
        done = step + 1
        row = {"step": done}
        if done % spec.log_every == 0:
            row["train_loss"] = sum(train_losses[-spec.log_every:]) / spec.log_every
            row["lr"] = lr
        if done % spec.eval_every == 0 or done == spec.steps:
            final_val = evaluate(model, windows)
            row["val_bpc"] = final_val
            diag.append({"step": done, "layers": diagnostics(model, windows)})
        if len(row) > 1:
            rows.append(row)
        if stop_hook is not None:
            stop_hook(done)
        if done < spec.steps and clock() - last_ckpt >= ckpt_seconds:
            atomic_torch_save({"spec": asdict(spec), "step": done, "model": model.state_dict(),
                               "opt": opt.state_dict(), "rng": torch.get_rng_state(),
                               "train_losses": train_losses, "rows": rows, "diag": diag,
                               "elapsed": elapsed + clock() - t0, "resumes": resumes}, ckpt_path)
            last_ckpt = clock()

    diverged = is_diverged(loss_ok, final_val, corpus.vocab_size)
    result = {
        "arm": spec.arm, "D": None if spec.arm == "uniform" else spec.head_dim,
        "seed": spec.seed, "stage": spec.stage, "lr": spec.lr, "steps": spec.steps,
        "final_val_bpc": final_val if loss_ok else None,
        "final_train_loss": (sum(train_losses[-FINAL_LOSS_WINDOW:]) / len(train_losses[-FINAL_LOSS_WINDOW:])
                             if train_losses else None),
        "diverged": diverged,
        "diverged_reason": None if not diverged else
                           ("non-finite training loss" if not loss_ok else "final bpc above log2(vocab)"),
        "steps_done": len(train_losses),
        "wall_time_s": elapsed + clock() - t0, "resumes": resumes,
    }
    if loss_ok:
        atomic_torch_save({"spec": asdict(spec), "model": model.state_dict()}, run_dir / "final.pt")
    atomic_write_text(run_dir / "log.csv", _log_csv(rows))
    atomic_write_json(run_dir / "diagnostics.json", diag)
    atomic_write_json(result_path, result)        # written last: it marks the run as finished
    if ckpt_path.exists():
        ckpt_path.unlink()
    log(f"{run_dir.name}: finished, val bpc {result['final_val_bpc']}"
        + (" (diverged)" if diverged else ""))
    return result
