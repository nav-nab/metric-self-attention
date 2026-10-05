# Version 1: design record

This file records the design of version 1 as it stood when the project was shelved. Each decision is marked **[user]** (stated by the
user, so it is the authority) or **[Claude]** (an implementation choice, not confirmed). The last column of each group says whether the
decision still holds for the restarted project (`docs/project_specification.md`). The recall task has its own record in `mqar/DESIGN.md`.

## Decisions that still hold

- **[user]** The score is the negative L^p distance between query and key. The grid is p in {1, 1.5, 2, 3, 4, 8, infinity}, with the
  plain distance (the p-th root is taken), including for p = 2. Learned p, SNCF and learned metrics are out of scope.
- **[user]** Queries and keys are tied (W_Q = W_K) in every arm except one untied dot-product reference arm, which measures the cost of tying.
  The tied dot product is the control of the crossed design. The untied arm has one extra projection matrix, and the write-up says so.
- **[user]** The diagonal is masked in the language model, for all arms, and position 0 may attend to itself. The reasoning was reviewed
  and confirmed on 2026-10-05 (see the Notes in the specification).
- **[user]** The logit is -beta x c x distance for distance arms and beta x c x (q . k) for dot arms. Here beta is a learned scale per
  head, with initial value 1 in every arm, and c is a frozen constant per arm.
- **[user]** The constant c is calibrated at initialisation so that the standard deviation of the logits across the allowed keys equals 1.
  Calibration is redone for every head size. Matching at initialisation is all that is done. The scale is not tuned per arm, and the
  write-up must say so. (The target of 1 was settled on 2026-10-05, after the pilot, which had treated it as provisional.)
- **[user]** The epsilon inside the root is relative: eps = r x (typical sum inside the root), measured at initialisation and frozen.
  p = infinity uses the maximum coordinate difference, with the gradient split evenly among tied coordinates, and no smoothing.
- **[user]** No normalisation of queries and keys. Learned absolute positions (no RoPE). The dependence of non-L2 metrics on the
  coordinate axes is accepted and flagged in the write-up.
- **[user]** The language model has 2 pre-norm decoder layers, d_model 64, 4 heads, MLP width 4 x d_model with GELU, context 64, tied
  input and output embeddings, on Tiny Shakespeare split 90/10. Validation bits per character is the outcome.
- **[user]** Arms are paired by seed: the same data stream and the same starting values for shared parameters.
- **[user]** The final weights of every run are saved. All benchmarks and checks are reachable from one entry script.
- **[user]** The compute budget is about 48 hours on a 2020 Intel MacBook. The distance kernel is a custom autograd function (chunked
  forward, analytic backward), with the naive version kept as a float64 reference.

## Decisions that have changed

- **Two tasks to one task.** Version 1 ran the recall task (MQAR) and the language model. The recall task is now shelved
  (`mqar/`), and the language model runs alone.
- **A single head size to a grid.** Version 1 used heads of dimension 16 only. The restarted project varies only the query-key
  dimension, over D = 2, 4, 8, 12 and 16, with the value width fixed at 16 per head, and adds a uniform-attention control arm.
- **The primary outcomes.** Version 1 had a recall outcome (the pair count P at which accuracy falls below 90%) and validation bpc for
  the language model. The recall outcome could not be computed for the L^p arms (see `mqar/PILOT_FINDINGS.md`).
- **The target spread** went from provisional to settled at 1, and calibration became per head size.

## Implementation choices by Claude (none approved)

- beta = exp(log_beta) per head. One calibrated constant per task configuration, arm and layer, measured on a calibration-seed model with
  fixed batches and shared by all seeds. At least 8 allowed keys per query enter the spread.
- AdamW (0.9, 0.99), weight decay 0.01 on matrices only, 100 warm-up steps, cosine decay to 10%, gradient clip 1.0, fp32.
  2500 steps with batch 64 for the language model. Learning rate 1e-3 by default. 3e-3 was proposed for the language model after a
  scan (see `PILOT_FINDINGS.md`), and the user has not decided it.
- Linear weights are drawn N(0, 1/fan_in), with residual output projections scaled by 1/sqrt(number of layers). Embeddings are drawn
  N(0, 0.02). Each parameter has its own generator, seeded by the run seed and the parameter name.
- Kernel chunk size 4, with causal block skipping. Checkpoints every 250 steps, bit-exact resume.
- **r = 1e-12 for the relative epsilon** and **the direct difference kernel for p = 2** (the matrix-multiply expansion lost accuracy at
  D = 1 and is opt-in). Both are awaiting the user's approval.

## Open decisions at the time of shelving

Approve r = 1e-12; approve the direct p = 2 kernel; the language-model learning rate and seed count; laptop or Modal; context 64 or 128;
whether the uniform control is a full arm; whether to commit weights to git; whether to run the first small-head learning-rate scan.
The head-size grid was settled on 2026-10-05 as D = 2, 4, 8, 12 and 16, and the nine-arm grid was confirmed.
