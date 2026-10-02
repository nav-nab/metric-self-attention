# Metric self-attention

Does replacing the query-key dot product in self-attention by a negative L^p distance score change what a
transformer can learn? Queries and keys are tied (W_Q = W_K) in every arm, and the score is
`-beta * c_p * ||q - k||_p` for p in {1, 1.5, 2, 3, 4, 8, infinity}. Controls: tied dot product (the crossed-design
control) and an untied dot product (the standard transformer, a reference arm outside the crossed design).

Two tasks:
* **MQAR variant A** (synthetic multi-query associative recall, one layer, one head, no MLP, no positions).
* **Character-level language model** on Tiny Shakespeare (2 layers, d_model 64, 4 heads of dimension 16).

The design decisions live in the project guide kept with the research notes (not in this repository).
This repository contains the code, the pilot, and the scripts for the full runs.

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate        # Python 3.10-3.12
pip install -r requirements.txt                               # torch 2.2.2 (last macOS Intel wheels), numpy<2
python execute_project.py env                                 # environment check + smoke training step
python execute_project.py pilot                               # the whole pilot, stage by stage (results/pilot/*.json)
python execute_project.py full --config configs/full.json --workers 2     # full experiment, resumable
python execute_project.py tests                               # unit tests
```

Everything (benchmarks, checks, calibration, pilot, full runs) is reachable from `execute_project.py`.
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
execute_project.py      single entry point
msa/lp.py               L^p distance kernel (naive / checkpointed / custom autograd / matmul for p=2, causal-block skipping)
msa/gradcheck.py        float64 gradient checks against naive autograd
msa/mqar_gen.py         MQAR variant A generator (vectorised + reference) and checks
msa/models.py           score module (beta, calibrated c, relative epsilon), MQAR model, character LM, shared-init rule
msa/calibration.py      empirical calibration of c and epsilon (diagonal masked)
msa/train.py            training loop, checkpointing, diagnostics, final weights
msa/evaluate.py         accuracy/bpc and attention diagnostics (entropy, margins, hub measure, ...)
msa/bench.py            timing and memory (one subprocess per measurement)
msa/pilot.py            pilot stages;  msa/full.py  full-run manifest
tests/                  pytest tests
data/tinyshakespeare.txt
results/pilot/          pilot outputs
```
