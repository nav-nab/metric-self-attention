# Metric self-attention

Does replacing the query-key dot product in self-attention by a negative L^p distance score change what a
transformer can learn? Queries and keys are tied (W_Q = W_K) in every arm, and the score is
`-beta * c_p * ||q - k||_p` for p in {1, 1.5, 2, 3, 4, 8, infinity}. Controls: tied dot product (the crossed-design
control) and an untied dot product (the standard transformer, a reference arm outside the crossed design).

The main experiment is a **character-level language model** on Tiny Shakespeare (2 layers, d_model 64, 4 heads, context 64).
The project studies how the head size D (the query-key dimension per head) interacts with the choice of p. Only the query-key
width is shrunk, and the value width stays at 16 per head. A uniform-attention control (every allowed key gets the same weight)
shows how much attention can move the result at all. See `docs/project_specification.md`.

The synthetic recall task (MQAR) is **shelved**: its code, design record and pilot findings are in `shelved/mqar/`. It may come back
after the full language-model results have been seen.

The design decisions live in the project guide kept with the research notes (not in this repository).
This repository contains the code, the pilot, and the scripts for the full runs.

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate        # Python 3.10-3.12
pip install -r requirements.txt                               # torch 2.2.2 (last macOS Intel wheels), numpy<2
python execute_project.py env                                 # environment check + smoke training steps
python execute_project.py pilot                               # the language-model pilot checks, stage by stage (results/pilot/*.json)
python execute_project.py full --config configs/full.json --workers 2     # full experiment, resumable
python execute_project.py tests                               # unit tests (main and shelved)
python shelved/mqar/execute_mqar.py -h                        # the shelved recall task
```

Everything for the language model (benchmarks, checks, calibration, pilot, small-head runs, full runs) is reachable from `execute_project.py`.
Run `python execute_project.py -h` for the list of stages. Stages that train many models take `--workers N`
(parallel processes, one thread each by default; on a 2-core machine two single-thread processes gave about
30% more throughput than one two-thread process).

## Interrupting and resuming

Every training run writes a checkpoint (model, optimiser, step, data-stream RNG state) every 250 steps and
resumes from it bit-exactly. Re-issue the same command to continue; finished runs are skipped.
Each run directory holds `config.json`, `log.jsonl`, `result.json` and `final.pt` (final weights, config,
calibration constants). `final.pt` is saved for every run, including pilot runs.

## Layout

```
execute_project.py      single entry point (language model)
conftest.py             puts the repository root on sys.path for the tests
msa/lp.py               L^p distance kernel (naive / checkpointed / custom autograd / matmul for p=2, causal-block skipping)
msa/gradcheck.py        float64 gradient checks against naive autograd
msa/arms.py             arm names and parsing (nine arms plus the uniform control)
msa/models.py           score module (beta, calibrated c, relative epsilon), character LM with a settable query-key dimension, shared-init rule
msa/calibration.py      empirical calibration of c and epsilon (diagonal masked)
msa/train.py            task-agnostic training engine (Task plug-ins), the language-model task, checkpointing, diagnostics, final weights
msa/evaluate.py         bpc and attention diagnostics (entropy, margins, ...)
msa/bench.py            timing and memory (one subprocess per measurement)
msa/pilot.py            pilot stages;  msa/full.py  full-run manifest;  msa/runner.py  parallel run driver
tests/                  pytest tests
shelved/mqar/           the shelved recall task: README, DESIGN, PILOT_FINDINGS, code, its own entry point and tests
docs/                   specification (current and original), pilot report (a snapshot), implementation notes
data/tinyshakespeare.txt
results/pilot/          pilot outputs (including the original MQAR pilot outputs)
```
