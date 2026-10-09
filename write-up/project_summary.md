# Metric self-attention: summary of the project, the design and the run

Status: working summary, written 2026-10-08 at the start of the analysis stage. It is not the write-up. It contains no results.

## 1. Question

The project replaces the query-key dot product in self-attention by a negative L^p distance. It asks how the outcome depends on the head size D, the dimension of the query and key vectors in each head. For each D, the questions are which p works best and whether any p beats the dot product.

The motivation is geometric. A small head limits the dot product more than the distance. The dot product can single out only keys at the edge of the key cloud. A distance can single out a key anywhere. The expected pattern is a distance advantage that is largest at small D and fades as D grows. This is a hypothesis for the experiment to test.

## 2. Design

**Model and data.** The model is a decoder-only transformer with two layers, width 64, four heads, an MLP of width 4 x 64 and a context of 64 characters. It uses learned absolute positions and tied input and output embeddings. The data are Tiny Shakespeare, split 90/10. The outcome is the final validation bits per character (bpc) after 4,000 steps.

**Head size.** Only the query-key dimension changes, with D in {2, 4, 8, 12, 16}. The value width stays at 16 per head.

**Arms.**
- Tied dot product (the control).
- Seven L^p distances, with p in {1, 1.5, 2, 3, 4, 8, infinity}.
- Untied dot product, with separate query and key matrices (a reference).
- Uniform attention (a descriptive control, outside the tests).
- Unmasked untied dot product at D = 16 only (the standard transformer, a reference).

The crossed arms give 45 cells. The two extra arms add 2 cells, so there are 47 cells. Each cell has five seeds, which gives 235 main runs.

**Score.** The tied arms share one projection for queries and keys. The score of a distance arm is minus the distance, scaled by a learned factor per head (initial value 1) and a frozen constant. The constant is calibrated at initialisation so that the scores have standard deviation 1 across the allowed keys. The scale is not tuned per arm.

**Masking.** The diagonal is masked in all arms except the unmasked reference. Position 0 may attend to itself. The dot-product baseline is therefore slightly non-standard.

**Learning rate.** A one-seed scan (seed 100) at 3e-3, 1e-2 and 3e-2 chose the rate of each cell. An extra rate of 1e-3 was run for the cells whose best base rate was 3e-3. The rates were chosen on the validation set that the results are reported on.

**Statistical plan.** The seeds are blocks, so comparisons are paired.
- Test 1 (primary): repeated-measures F-test of the D x p interaction over the seven L^p arms.
- Test 3 (primary): trend of the paired gap to the tied dot product across log2 D.
- Test 2 (secondary): paired gap per cell, with Holm correction within each D.
- Test 4 (secondary): linear and quadratic contrasts across 1/p.
- Test 5 (secondary): bootstrap probability that each p is best.
- Test 6 (equivalence): not coded. Its margin is not chosen. It is needed only if the write-up claims equivalence.

The uniform control and the two untied arms enter no test. No permutation tests are used.

## 3. The run

The full run finished on 2026-10-07 on the user's laptop. It has 187 scan runs and 235 main runs. No run diverged. The scan chose the rate 3e-3 for 46 cells and 1e-2 for the uniform control. Per-arm rate tuning is therefore weak, and the rate is close to a shared one. The median seed standard deviation is 0.0096 bpc. The user decided that no finer learning-rate scan is made. All conclusions rest on the existing run.

## 4. Statements the write-up must make

- The scale of the score was not tuned per arm, and the target spread was chosen by convention.
- The diagonal mask makes the baseline non-standard.
- The epsilon puts a floor under the distance, about 3 percent at p = 8.
- The learning rate was tuned per arm and head size on the reporting set.
- The untied arm has one extra projection matrix.
- L^p distances with p other than 2 depend on the coordinate axes.
- The conclusions rest on learned absolute positions.
- Any divergent arm is reported as a finding (none occurred).
