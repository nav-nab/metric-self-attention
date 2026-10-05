# Metric self-attention: implementation plan

This is the third document. It follows `project_outline.md` and `design_details.md`, which fix what is run. This document fixes how the code will be organised. It is a plan for the user to review. No code exists yet, and nothing is run until the user agrees.

Items marked **Proposal** are my suggestions and need the user's approval. Items marked **Requirement** were requested by the user.

## 1. Scope

The whole project is one program that runs on the user's laptop. A single script, `execute_project.py`, performs every step: the tests, the calibration, the learning-rate scan, the main experiment and the analysis. The code is Python scripts, not notebooks. The script never runs git. The user commits the results and pushes them.

## 2. Layout

```
execute_project.py        the single entry point
msa/
  data.py                 corpus, split, batches, seeds
  model.py                the transformer and the arm switch
  scores.py               the score functions and the L^p kernel
  calibrate.py            the calibration constants
  train.py                one run, with checkpointing
  sweep.py                the scan, the rate selection and the main grid
  analyze.py              the statistical tests, tables and figures
tests/                    unit tests and the interruption test
results/<stage>/<run>/    result.json, log.csv, final.pt, ckpt.pt
docs/figures/             schematic figures (already present)
```

The run folder has two levels under `results/`, which matches the existing `.gitignore` rule that excludes `ckpt.pt` and keeps `final.pt`.

## 3. The execute script

One command runs everything, and each stage can also be run alone.

| Command | What it does |
|---|---|
| `check` | Runs the unit tests. |
| `smoke` | Runs every stage on a tiny scale (a few steps, two cells). **Requirement**: needs the user's agreement before it is run. |
| `calibrate` | Measures and saves the constants of section 6 of the details. |
| `scan` | Stage 1 of the learning-rate protocol: 141 runs. |
| `select` | Chooses the rate of each cell and lists the cells that need an extra rate. |
| `main` | Stage 2: 235 runs at the chosen rates. |
| `analyze` | Reads the results, writes the tables and the figures. |
| `all` | Runs the stages in order, including any extra scan runs. |

The stages depend on each other only through files. `select` reads the scan results, and `main` reads the selected rates. The script holds no other state, so any stage can be restarted.

## 4. Interruption and resumption

The full run takes about a day, so interruptions are expected. **Requirement**: the script must resume approximately where it left off.

- **Finished runs are skipped.** A run is finished when its `result.json` exists. Rerunning a stage starts only the runs that are not finished.
- **Checkpoint every 10 minutes.** **Requirement**: each running run saves a checkpoint every 10 minutes of wall-clock time. The checkpoint holds the model, the optimiser, the schedule position, the step count, the random-number states and the position in the data stream. A restarted run loads its checkpoint and continues from that step.
- **Writes are atomic.** A file is written under a temporary name and then renamed. An interruption during a write cannot leave a damaged checkpoint or result.
- **Cleaning.** A run deletes its checkpoint when it finishes. The checkpoint is never committed.
- **A limit worth knowing.** Many runs last less than 10 minutes. Such a run may never reach a checkpoint, and an interrupted run then restarts from step 0. The loss is under 10 minutes of work per worker.
- **Tested.** An interruption test stops a short run in the middle and resumes it. The resumed run must agree with the uninterrupted run to within floating-point noise.

## 5. Calibration and divergence

**Calibration.** The calibration stage fixes the points left open in section 6 of the details. The standard deviation is measured per query over its allowed keys. It is then pooled over the positions with at least about 8 allowed keys, over the four heads of the layer, and over the fixed calibration batches. The constants are saved to a file, and every later run reads that file.

**Divergence.** The script must decide without a person whether a run diverged. **Proposal**: a run diverged if its training loss becomes non-finite at any step, or if its final validation bpc is above $\log_2 65 \approx 6.02$, which is the value of a model that guesses uniformly over the characters. A diverged scan run is recorded, reported and excluded from the choice of rate. A diverged main run is recorded as such and is not rerun.

## 6. Training choices left to coding time

The details document leaves four choices open. **Proposal**:

| Choice | Proposed value |
|---|---|
| Optimiser | AdamW, with betas 0.9 and 0.95, and a small weight decay on the matrices only |
| Schedule | Linear warm-up over the first 200 steps, then cosine decay to 10 percent of the peak rate. The scan tunes the peak rate. |
| Batch | 64 sequences of 64 characters per step |
| Gradient clipping | Global norm 1.0, for every arm |
| Evaluation | Validation bpc on the whole validation text in non-overlapping windows of 64 characters, every 500 steps and at the end. The outcome is the final value. |

These values are standard defaults. They are the same for every arm, and only the peak rate is tuned. The user may change any of them.

## 7. Speed and parallelism

The estimates came from sandbox runs. About four workers run at once, and each worker uses one thread. **Proposal**: the sweep puts the slowest cells first, so that the last runs of a stage are short. The script prints a progress line with the number of finished runs and an estimate of the time remaining. The user should keep the laptop awake during the run, for example with `caffeinate -i` on macOS.

## 8. Analysis

The analysis reads only the result files. It needs a library for the mixed model of test 1. **Proposal**: add `statsmodels` to `requirements.txt`, at a version that works with torch 2.2.2, numpy below 2 and Python 3.10 to 3.12. The exact form of the F-test for the D x p interaction is written out in this document before the analysis code is written. Test 6 is not coded, because it is run only if an equivalence claim is made.

## 9. Tests

The unit tests check the following.
- The L^p kernel agrees with a plain float64 version in value and gradient, for every p including infinity.
- The gradient at an exact match is finite for every p, because of the epsilon.
- The mask is correct, including the exception at position 0.
- After calibration, the scaled scores of each arm have a standard deviation of 1 on the calibration batches, and the tied dot constant is near $1/\sqrt{D}$.
- Arms run with the same seed start from the same shared parameters.
- The data split and the batches are the same for the same seed.
- A short run resumes correctly after an interruption (section 4).

## 10. Order of work

1. The data, the model and the score functions, with their tests.
2. Calibration, with its tests.
3. One run and its checkpointing, with the interruption test.
4. The sweep stages, the rate selection and the `smoke` command.
5. A short smoke run on the user's laptop, after the user agrees, to catch problems specific to the platform before the long run.
6. The analysis code, tested on invented results and not on real runs.
7. The long run, started by the user.

## 11. For the user to decide

- Approve or change the proposals in sections 5, 6, 7 and 8.
- Agree to the smoke run in step 5 of section 10 before it is run.
