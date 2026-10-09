# Analysis findings (2026-10-09)

Record of the analysis stage (2026-10-09). The statistical tests listed here are tests 2 and 3 of the plan, with the untied and unmasked contrasts. Tests 1, 4 and 5 of the plan were not carried out, and they are not used in this record. The paper in `paper/` reports the same results.

## Integrity checks (all passed)
- 235 main and 187 scan runs, five identical seeds in every cell, no divergence, no NaN, no stray files.
- Every result.json equals the last row of its log.csv. The learning-rate column follows the schedule. Main rates equal selected_rates.json. Scan choices are the lowest-bpc rates.
- All 235 final.pt files reload and reproduce the stored validation bpc to about 1e-7. Stored c and eps match the calibration files in all 828 layers.
- The nine resumed main runs are not outliers (largest deviation 1.5 seed sd).
- Rerunning `analyze` leaves all 11 tables byte-identical. Tests 2 and 3 were recomputed independently (`paper/verify_tests.py`). The two agree up to the six-digit rounding of the CSV files.

## The three themes (agreed 2026-10-09)
1. The dot product is the better score in this setting.
2. The surface is flat, and p matters little.
3. Costs and one time-dependent pattern: run time, and an early advantage of the L^p arms that disappears.

## Theme 1: approved tests and results
Common setup: the seed is the paired block. The gap is g = bpc(L^p arm) - bpc(tied dot), same seed and D. Positive means the L^p arm is worse. All t-tests have 4 degrees of freedom.

- Claim A (cell by cell, test 2, Holm within each head size over seven arms): the mean gap is positive in 34 of 35 cells. The one exception is p = 2 at D = 16 (-0.005, not significant). After Holm, 15 of 35 cells are significant: 7 of 7 at D = 2, 3 at D = 4, 2 at D = 8, 2 at D = 12 and 1 at D = 16. No cell favours an L^p arm. D = 2 gaps are 0.036 to 0.050 bpc. Gaps at larger D are mostly 0.01 to 0.02 bpc.
- Claim B (untied minus tied dot, five uncorrected rows of test 2): the mean is negative at all five head sizes (-0.004, -0.0065, -0.011, -0.005, -0.013 bpc). Only D = 8 has p < 0.05 uncorrected (p = 0.012). D = 16 has p = 0.069.
- Claim C (unmasked minus masked untied dot, D = 16): +0.0033 bpc, 95% interval -0.0132 to +0.0198, p = 0.61. There is no detectable cost of masking. The interval does not exclude differences of 0.02 bpc. Test 6 is not run.
- Claim D (test 3, slope of the gap on log2 D, per seed and arm, one-sample t-test, no multiplicity correction by the user's earlier decision): the mean over the seven arms has slope -0.0086 bpc per doubling of D (95% interval -0.0127 to -0.0044, t = -5.70, p = 0.0047), and all five seed slopes are negative. All eight rows have negative mean slopes. Five arms have p < 0.05 (p = 1.5, 2, 3, 4, 8). The arms p = 1 and p = infinity do not (p = 0.22 and 0.25). The hypothesis predicted a positive slope. The result is the opposite sign.

## Descriptive results (no tests)
- Range: all 45 crossed cells lie between 2.325 and 2.394 bpc. Uniform attention is 2.929.
- Tied dot product: 2.344 (D = 2) to 2.338 (D = 16), nearly flat.
- Spread across p at one D: 0.014 bpc at D = 2.
- Early advantage (descriptive mention only, by the user's decision): at step 500 the L^p arms are ahead of the tied dot in 33 of 35 cells (mean lead 0.027 bpc, largest 0.054). The mean gap is about zero at step 1000 and +0.021 at step 4000.
- Run time (one statement and one table in the write-up, by the user's decision): dot arms 7 to 8.5 minutes per run at every D. L^p arms 10 to 49 minutes, growing with D. Whole-run ratios to the tied dot are 1.3 to 1.9 at D = 2 and 2.6 to 5.8 at D = 16. This is a chunked PyTorch kernel on laptop CPUs, four runs in parallel.
- Learning rates: 46 of 47 cells chose 3e-3. The uniform control chose 1e-2.

## Not done
- Tests 1, 4 and 5 (interaction, shape across 1/p, probability of being best) are not opened. They await discussion with the user.
