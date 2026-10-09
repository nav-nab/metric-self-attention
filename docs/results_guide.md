# Metric self-attention: results guide

This is the fourth document. Read `project_outline.md`, `design_details.md` and `implementation_plan.md` first. They fix the question, the design and the code. This document describes what the run produced and where to find it. 

The document states facts about the run and about the files. It does not interpret the results. The interpretation is in `analysis_findings.md` and in the paper.

**Contents.**
1. Status of the project
2. Record of the run
3. Checks of the run that were already made
4. Where the files are and what they contain
5. How the analysis was defined
6. Points to keep in mind when interpreting the results

## 1. Status of the project

The design is settled. The code is written and tested. The full experiment has finished on the user's laptop. All 235 main runs and all 187 scan runs are complete, and no run diverged. The analysis stage has been run on the real results. Its tables and figures are in `results/analysis/`.

The code that produced the results is the commit `b4ade25` and its predecessors. All 150 unit tests passed on that commit. They were run on Linux with Python 3.10 and torch 2.2.2. The user ran the smoke test and the long run on the laptop, and the user starts every stage. The script never runs git.

The results folder `results/` holds about 175 MB, and most of this is the final weights, at about 0.4 MB per run. The results are committed to git, including the final weights of the scan and main runs. Checkpoints and the smoke run are not tracked.

## 2. Record of the run

All file times below are UTC times of the result files on the laptop.

| Stage | Period | Runs | Comment |
|---|---|---|---|
| Calibration | 5 Oct, 13:23 to 13:25 | 46 cells | one file per arm and head size |
| Scan | first result 5 Oct, 17:49; last result 6 Oct, 11:18 | 187 runs | 141 base runs and 46 extra runs |
| Main | 6 Oct, 18:46 to 7 Oct, 19:11 | 235 runs | five seeds in each of 47 cells |
| Analysis | 7 Oct, 19:12 | | `results/analysis/` |

The elapsed time from the first scan result to the analysis was about 49 hours. This includes pauses between the stages. The sum of the wall-clock times of the single runs was 66.9 hours for the scan and 81.4 hours for the main stage. With four workers this is about 37 hours of pure compute. The plan had estimated 23 to 24 hours. The mean run took 21 minutes, and the plan had assumed about 14 minutes.

The mean wall-clock time of one main run, in minutes, was as follows. The scan runs of the same arm differ by less than 10 percent.

| Arm | Minutes | Arm | Minutes |
|---|---|---|---|
| uniform | 4.5 | lp_3 | 25.2 |
| dot_tied, dot_untied | 7.8 | lp_inf | 27.4 |
| dot_untied_unmasked | 8.2 | lp_1.5 | 27.6 |
| lp_1 | 16.5 | lp_4 | 26.8 |
| lp_2 | 22.1 | lp_8 | 31.6 |

The run was interrupted at least once. Six scan runs and nine main runs were resumed from a checkpoint. The field `resumes` in `result.json` counts the resumptions of each run. The tests show that a resumed run agrees with an uninterrupted run to better than 1e-6 in the final bpc. The nine resumed main runs are not outliers (`analysis_findings.md`).

**Learning-rate scan.** The scan used the seed 100. The base grid was 3e-3, 1e-2 and 3e-2. The extra rate 1e-3 was run for the 46 cells whose best base rate was 3e-3. The extra rate 9e-2 was never needed. The chosen rate was 3e-3 in all 46 cells other than the uniform control. The uniform control chose 1e-2.

The rate 3e-3 beat 1e-3 by 0.063 to 0.141 bpc, with a median of 0.093. It beat 1e-2 by 0.003 to 0.087 bpc, with a median of 0.036. In five cells the margin over 1e-2 was below 0.01 bpc, which is about the seed noise. The full scan table is `results/scan_table.csv`.

**Seed noise.** The standard deviation of the final bpc over the five seeds of a cell has a minimum of 0.0034, a median of 0.0096 and a maximum of 0.0267. The values per cell are in `cell_means.csv`.

## 3. Checks of the run that were already made

After the run, the following checks were made on the laptop files.
- There are 187 scan folders and 235 main folders, and each has a `result.json`.
- Each run folder holds `result.json`, `log.csv`, `diagnostics.json` and `final.pt`.
- No `ckpt.pt`, `progress.json`, `error.txt` or temporary file is left outside `results/smoke/`.
- The rate in each main `result.json` equals the rate in `selected_rates.json`.
- `selected_rates.json` is complete. No cell needs an extra rate, no cell is missing, and no cell diverged at every rate.
- The table `diverged_runs.csv` is empty.

The following checks were made later, and all passed. They are recorded in `analysis_findings.md`.
- A check that the validation bpc in each `log.csv` ends with the value in `result.json`.
- A check of the stored weights, for example by reloading one `final.pt` and recomputing its validation bpc.
- A comparison of the calibration constants with the stored statistics of the weights after training.

## 4. Where the files are and what they contain

All paths are relative to the repository folder. The arm names are `dot_tied`, `dot_untied`, `dot_untied_unmasked`, `uniform`, and `lp_1`, `lp_1.5`, `lp_2`, `lp_3`, `lp_4`, `lp_8`, `lp_inf`. A cell identifier is the arm and the head size, for example `lp_1.5_D4`. The uniform control is the cell `uniform`.

**Run folders.** Scan runs are in `results/scan/<cell>_lr<rate>/`. Main runs are in `results/main/<cell>_seed<seed>/`. Each folder has these files.

| File | Content |
|---|---|
| `result.json` | `arm`, `D` (null for the uniform arm), `seed`, `stage`, `lr`, `steps`, `final_val_bpc`, `final_train_loss`, `diverged`, `diverged_reason`, `steps_done`, `wall_time_s`, `resumes` |
| `log.csv` | columns `step`, `train_loss`, `lr`, `val_bpc`. The training loss is the mean over the last 50 steps, in nats. The validation bpc is in bits per character and is given every 500 steps. |
| `diagnostics.json` | a list with one entry per evaluation, each with `step` and `layers`. Each layer has per-head lists `beta`, `max_logit`, `entropy`, `q_norm` and `k_norm`. The uniform arm has no diagnostics. |
| `final.pt` | the dictionary with `spec` (the run settings) and `model` (the state dict, including the frozen constants `c` and `eps` of each attention layer) |

The final training loss is the mean over the last 100 steps and it is in nats. The final validation bpc is the value at step 4000. It is not the best value during training.

**Calibration.** `results/calibration/<arm>_D<D>.json` holds, for each layer, the constant `c`, the measured spread `sigma`, the typical sum `s_typ` and the epsilon `eps`. It also holds the number of pooled queries, the spreads of the even and odd batches, and the variance `z_var` of the coordinates of the query. `results/calibration/calibration_summary.csv` collects all of these in one table. The column `c_zvar_sqrtD` is the check of the dot-product constant that plan section 13, item 1, describes.

**Selection.** `results/selected_rates.json` holds, for each cell, the chosen rate, its bpc and the scan results at each rate. `results/scan_table.csv` has one row per scan run with the columns `cell`, `lr`, `val_bpc`, `diverged` and `chosen`.

**Analysis tables.** They are in `results/analysis/tables/` and are written by `python execute_project.py analyze`.

| Table | Content |
|---|---|
| `results_long.csv` | one row per main run: cell, arm, D, seed, lr, val_bpc, diverged, diverged_reason, final_train_loss |
| `cell_means.csv` | one row per cell: lr, n_valid, mean_bpc, sd_bpc, min_bpc, max_bpc |
| `chosen_rates.csv` | the chosen rate and the scan bpc at each of the five rates |
| `diverged_runs.csv` | the diverged main runs. It is empty. |
| `test1_interaction.csv` | the F-test of the D x p interaction: F, df1, df2, p, the two sums of squares and the removed arms |
| `test2_gaps.csv` | the paired gap of each $\ell^p$ arm to the tied dot product at each head size: mean, 95 percent interval, t, p, Holm-adjusted p and the five single-seed gaps. It also has the rows for untied minus tied and unmasked minus masked, which are labelled uncorrected. |
| `test3_trend.csv` | the slope of the gap per doubling of D, for each arm and for the mean of the seven arms: mean, interval, t, p and the five single-seed slopes |
| `test4_shape.csv` | the linear and quadratic contrasts along 1/p at each head size, and their slopes on log2 D |
| `test5_best_probability.csv` | for each head size, the probability that each p has the lowest mean bpc |
| `test5_rank_correlation.csv`, `test5_rank_stability.csv` | the rank correlations of the ordering of p between head sizes, and their summary |

**Figures.** They are in `results/analysis/figures/`: `bpc_by_head_size.png`, `gap_by_head_size.png`, `shape_over_p.png` and `best_probability.png`.

**How to rerun the analysis.** The command `python execute_project.py analyze` reads only the result files and takes under a minute. It does not train anything. The command `python execute_project.py check` runs the unit tests and takes about three minutes.

## 5. How the analysis was defined

The tests are defined in `design_details.md`, section 9. The exact form of test 1 is in `implementation_plan.md`, section 8.1. The code choices for the other tests are items 23 to 30 of plan section 13. The main points follow.

- The outcome is the final validation bpc. A lower value is better. The seed is a block.
- The gap of an $\ell^p$ arm is its bpc minus the bpc of the tied dot product with the same seed and head size. A negative gap means that the distance arm is better.
- Test 1 is the repeated-measures F-test of the D x p interaction, with 24 and 96 degrees of freedom. It has no correction for sphericity. This was decided by the user.
- Test 2 gives intervals without a correction and p-values with a Holm correction over the seven arms at one head size.
- Test 3 has no correction for multiple testing across its eight rows. This was decided by the user. The mean over the seven arms is the primary row.
- Test 5 enumerates all 126 resamples of the seeds, so it uses no random numbers.
- Test 6, the equivalence test, is not coded and was not run, because the paper makes no equivalence claim. Its margin was never chosen.
- The paper reports tests 2 and 3 and the untied and unmasked contrasts. Tests 1, 4 and 5 are computed by the analyze stage but are not reported.

## 6. Points to keep in mind when interpreting the results

These points come from the design and from the run. Each can affect how a result should be read.

1. **The learning rates are almost the same in all cells.** The scan chose 3e-3 for 46 of the 47 cells. The grid has steps of a factor of about 3, and the best rate lies between 1e-3 and 1e-2. The claim that the rate was tuned per arm is therefore weak. Differences between arms may still depend on how far each arm's own optimum is from 3e-3. The scan table shows the bpc at the neighbouring rates. No finer scan was made. All conclusions rest on the run that exists.
2. **The rates were chosen on the validation set that the results are reported on.** The scan used a seed that is not one of the five final seeds. The paper says this.
3. **Calibration matches the spread of the scores at initialisation only.** The learned scale beta and the weights change during training. The dot product depends quadratically on the scale of the query vectors, and a distance depends linearly on it. The diagnostics hold the learned scale, the largest logit, the entropy and the norms of the queries and keys, so that any later drift can be examined.
4. **The epsilon is far below the float32 resolution of the typical sum.** It acts only for near-exact matches. The floors that the design lists (about 3 percent at p = 8) hold for the distance at an exact match and not for typical pairs.
5. **The baseline is non-standard in two ways.** The diagonal is masked, and the query and key matrices are tied. The arm `dot_untied_unmasked` at D = 16 measures the effect of the mask together with the untying. The arm `dot_untied` measures the effect of tying.
6. **At D = 2 the arms p = 1 and p = infinity have the same expressive power.** Any difference between them at D = 2 reflects optimisation.
7. **The design makes no claim about underfitting.** The final training loss in `result.json` is in nats, and the validation bpc is in bits. For example, the cell `dot_tied_D2` seed 1 has a training loss of 1.428 nats, which is 2.060 bits, and a validation bpc of 2.352. The gap between the two is about 0.3 bits. Any statement about fit needs a check of all cells first.
8. **Five seeds give wide intervals.** The smallest detectable difference is about 0.02 bpc without a correction, and about 0.035 bpc with the Holm correction. The single-seed values are in the tables and should be shown with every summary.
9. **The uniform control is a descriptive scale only.** It is not entered into any test. The two untied arms are not part of test 1, 3, 4 or 5.
11. **The scan runs are not a sixth seed.** They use the seed 100 and they were part of the selection of the rates. They must not be pooled with the main results.
