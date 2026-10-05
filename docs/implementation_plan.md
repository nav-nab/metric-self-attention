# Metric self-attention: implementation plan

This is the third document. It follows `project_outline.md` and `design_details.md`, which fix what is run. This document fixes how the code is organised. It is the brief for the agent that writes the code. The code is being written, and no experiment is run until the user agrees.

Items marked **Requirement** were requested by the user. Items marked **Approved** were proposed by Claude and approved by the user on 2026-10-05. Section 11 lists details that the documents leave to the implementing agent, together with the rules for working on this project.

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
results/<stage>/<run>/    result.json, log.csv, diagnostics.json, final.pt, ckpt.pt
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

**Divergence.** The script must decide without a person whether a run diverged. **Approved**: a run diverged if its training loss becomes non-finite at any step, or if its final validation bpc is above $\log_2 65 \approx 6.02$, which is the value of a model that guesses uniformly over the characters. A diverged scan run is recorded, reported and excluded from the choice of rate. A diverged main run is recorded as such and is not rerun.

## 6. Training choices left to coding time

The details document left these choices to coding time. **Approved**:

| Choice | Value |
|---|---|
| Optimiser | AdamW, with betas 0.9 and 0.95, and a small weight decay (0.01) on the weight matrices only |
| Schedule | Linear warm-up over the first 200 steps, then cosine decay to 10 percent of the peak rate. The scan tunes the peak rate. |
| Batch | 64 sequences of 64 characters per step |
| Gradient clipping | Global norm 1.0, for every arm |
| Evaluation | Validation bpc on the whole validation text in non-overlapping windows of 64 characters, every 500 steps and at the end. The outcome is the final value. |

These values are standard defaults. They are the same for every arm, and only the peak rate is tuned. The user may change any of them.

## 7. Speed and parallelism

The estimates came from sandbox runs. About four workers run at once, and each worker uses one thread. **Approved**: the sweep puts the slowest cells first, so that the last runs of a stage are short. The script prints a progress line with the number of finished runs and an estimate of the time remaining. The user should keep the laptop awake during the run, for example with `caffeinate -i` on macOS.

## 8. Analysis

The analysis reads only the result files. It needs a library for the mixed model of test 1. **Approved**: add `statsmodels` to `requirements.txt`, at a version that works with torch 2.2.2, numpy below 2 and Python 3.10 to 3.12. The exact form of the F-test for the D x p interaction is written out in a short section of this document before the analysis code is written, and the user approves it. The two usual forms are a mixed model with a random seed intercept and a repeated-measures analysis of variance. Test 6 is not coded, because it is run only if an equivalence claim is made.

## 9. Tests

The unit tests check the following.
- The L^p kernel agrees with a plain float64 version in value and gradient, for every p including infinity.
- The gradient at an exact match is finite for every p, because of the epsilon.
- The mask is correct, including the exception at position 0.
- After calibration, the scaled scores of each arm have a standard deviation of 1 on the calibration batches. The tied dot constant satisfies $c\,\sigma_z^2\sqrt{D} \approx 1$, where $\sigma_z^2$ is the measured variance of the coordinates of $z$ (section 13, item 1).
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

## 11. Further details and working rules for the implementing agent

**Details the documents do not fix.** Section 11 gives defaults. The agent may choose another standard value. It then records the choice in section 13 and tells the user, and it flags any choice that could affect the conclusions.

- **Data.** The vocabulary is the sorted set of characters of the whole corpus. The training text is the first 90 percent of the characters, and the validation text is the last 10 percent. A training batch is made of random windows of 65 characters, drawn from the training text by a generator that depends on the seed. The inputs are the first 64 characters and the targets are the last 64. The validation windows do not overlap, and a final partial window is dropped.
- **Seeds.** The five final seeds are 1 to 5. The scan seed is 100. The calibration seed is 999. The data stream and the initial values use separate generators for each seed.
- **Shared initial values.** Each parameter is initialised from a generator seeded by the seed and the parameter name. Arms with the same seed then share the initial values of every parameter they have in common. In the untied arm, the initial $W_Q$ equals the initial $W$ of the tied arms, and $W_K$ comes from its own generator.
- **Initialisation.** Matrices and embeddings are drawn from a normal distribution with standard deviation 0.02. Norm gains start at 1. The output projections of the two residual branches are scaled by $1/\sqrt{2L}$, with $L = 2$ layers.
- **Layers.** Pre-normalisation with LayerNorm, a final LayerNorm before the tied output projection, GELU in the MLP, and no dropout. The query, key and value projections have no bias, because a shared bias cancels in a distance.
- **The scale $\beta_h$.** It is stored as $\log\beta_h$ with initial value 0, so that the scale stays positive. It has no weight decay.
- **Weight decay.** It applies to the weight matrices of the linear layers only. It does not apply to embeddings, norm gains, biases or $\log\beta_h$.
- **Calibration batches.** A fixed set of batches drawn with the calibration seed. The number is chosen so that the constants are stable to about 1 percent.
- **The uniform control.** It has no query or key projections. All other parts of the model are the same.
- **Numbers.** Everything is float32. Each worker uses one thread and the multiprocessing method `spawn`.
- **Result files.** Each run writes `result.json` with the arm, D, seed or scan flag, rate, number of steps, final validation bpc, final training loss, the diverged flag and the wall-clock time. The file format is otherwise the agent's choice.

**Working rules.**
- The design in `project_outline.md` and `design_details.md` is settled. The agent does not change it. If something looks wrong or ambiguous, the agent asks the user.
- The agent ignores the `shelved/` folder, and it does not use the version-1 code kept in the project memory.
- Unit tests may be run freely. A smoke run on the user's laptop, the learning-rate scan and the main experiment are run only when the user agrees. The user starts the long run.
- The agent commits locally and does not push. It keeps the three documents in step with the code, and it writes documents in formal prose with short, self-contained clauses.
- Complex extras are not added by default. The write-up has to be understood by human readers.

## 12. Items that still need the user

- Agreement to the smoke run in step 5 of section 10 before it is run.
- Approval of the written form of the test-1 F-test (section 8) before the analysis code is written. The user has chosen the repeated-measures analysis of variance.
- Review of any choice in section 13 that the agent flags.

## 13. Coding decisions (filled in by the implementing agent)

Each item records a choice that the documents leave open, or a detail of the code that the reader may need. Items marked **User** were decided by the user. The other items follow the defaults of section 11, or are small details of the code.

**Steps 1 and 2 (data, model, scores, calibration).**

1. **Sanity check of the tied dot constant. User.** With the initial standard deviation of 0.02 from section 11, the coordinates of $z$ have a standard deviation of about 0.16, not 1. The constant then lands near $40/\sqrt{D}$ and not near $1/\sqrt{D}$. The check of design section 6 is therefore restated as $c\,\sigma_z^2\sqrt{D} \approx 1$, with $\sigma_z^2$ measured on the calibration batches. The initial standard deviation stays at 0.02. The measured product lies between 0.84 (D = 16) and 1.02 (D = 2) for the tied dot arm. The test accepts 0.75 to 1.25.
2. **Pooling of the spread.** The variances of the raw scores of single queries are averaged over the pooled queries. The constant is the inverse of the square root of that mean variance. This is the root mean square of the per-query standard deviations.
3. **The typical sum $S_{\text{typ}}$.** It is the median, over the same pooled pairs of query and key, of $\sum_m |x_m - y_m|^p$. The median is used because the sums are skewed at large p. It is measured with the epsilon set to 0, so that it equals the p-th power of the distance exactly.
4. **Calibration batches.** There are 32 batches of 64 sequences, drawn with seed 999. The constants measured on the even and the odd batches agree to within 1 percent in the tests.
5. **Calibration files.** Each cell has its own file, `results/calibration/<arm>_D<D>.json`. A finished file is skipped on a restart. The stage also writes `calibration_summary.csv`. The constants are stored as `c` and `eps`. The epsilon is 0 for p = 1, for p = infinity and for the dot arms.
6. **Arm names.** `dot_tied`, `dot_untied`, `dot_untied_unmasked`, `uniform`, and `lp_1`, `lp_1.5`, `lp_2`, `lp_3`, `lp_4`, `lp_8`, `lp_inf`.
7. **Batches.** A training batch is a pure function of the seed and the step. A resumed run then sees the same batches as an uninterrupted run, and the checkpoint needs no position in the data stream beyond the step.
8. **Layers.** The output projection and the two MLP layers have biases. LayerNorm has a gain and a bias. The query, key and value projections have none, as section 11 says. The tied matrix is stored as `w_q` in every arm, so that the untied arm starts with the same `w_q` as the tied arms.
9. **The kernel.** It processes 16 query rows per chunk. In the causal mode it computes only the keys up to the end of the chunk, which halves the work. Entries above the diagonal are then not meaningful, and the model masks them. Squares, cubes and the exponents 4, 7 and 8 use multiplications, and the other exponents use `pow`. The backward pass sets the gradient to zero where the upstream gradient is zero. Masked entries then cannot produce 0 times infinity, even if the epsilon is 0.
10. **Extra module.** `msa/util.py` holds the atomic file writes. It is not in the layout of section 2.
11. **Environment.** The unit tests were run in a virtual environment on Linux with Python 3.10, torch 2.2.2 and numpy 1.26. The line `statsmodels>=0.14,<0.15` was added to `requirements.txt`. Version 0.14.6 imports with this torch and numpy. The environment lies outside the repo folder.

**Step 3 (one run and its checkpointing).**

12. **Files of a run.** `result.json` is written last, and it marks the run as finished. `log.csv` holds the training loss and the rate every 50 steps and the validation bpc every 500 steps. `diagnostics.json` holds the per-head diagnostics of section 3 of the details at every evaluation. A run with a non-finite loss stops at once and is recorded as diverged. It has no `final.pt`, because its weights are not finite. Its `result.json` has the validation bpc `null`.
13. **Diagnostics.** They are measured on the first 64 validation windows. The entropy is the mean over the queries with at least 8 allowed keys. The largest logit is the largest scaled score of the head. The norms are means over all positions. The uniform arm has no diagnostics.
14. **Checkpoint.** It holds the model, the optimiser, the next step, the loss history, the log rows, the diagnostics, the elapsed time and the torch random-number state. The training uses no random numbers beyond the seeded batches, so the state is kept only as the plan requires. A checkpoint of a different run (different arm, head size, seed, rate or length) is refused. A run that ends or fails before its first checkpoint restarts at step 0, as section 4 says.
15. **Numbers.** The final training loss is the mean over the last 100 steps. The rate rises linearly over the first 200 steps and reaches 10 percent of the peak exactly at the last step. The weight decay of 0.01 applies to the weight matrices of the linear layers (five per layer, six in the untied arms). The result of the uniform arm records the head size as `null`.
16. **Interruption tests.** One test stops a run with an exception after step 13 with a fake clock, so that the last checkpoint is at step 10. It resumes the run and compares the weights, the log and the final bpc with an uninterrupted run, for three arms. A second test kills a real process with SIGKILL while it writes a checkpoint at every step. It then resumes the run and compares the final bpc with an uninterrupted run. Both agree to better than 1e-6.
