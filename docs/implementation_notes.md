# Implementation notes (for the write-up)

These notes describe what the code in this repository actually does. Design decisions are in the project guide
(kept with the research notes). Numbers from the pilot are in `results/pilot/` and summarised in `docs/pilot_report.md`.

## Arms and the score

* Nine arms: `dot_tied`, `lp_1`, `lp_1.5`, `lp_2`, `lp_3`, `lp_4`, `lp_8`, `lp_inf`, `dot_untied`.
* Every arm uses one score module (`msa/models.py: Scorer`).
  Distance arms: `logit = -beta_h * c * dist_p(z_q, z_k)`; dot arms: `logit = beta_h * c * (z_q . z_k)`.
  `beta_h = exp(log_beta_h)` is learnable per head and starts at 1 in every arm. `c` is a frozen calibrated constant
  (a buffer, saved with the weights). Tied arms use `z = W u` for queries and keys; `dot_untied` has separate `W_q`, `W_k`.
* Distance: `dist = (S + eps)^(1/p)` with `S = sum_d |z_q,d - z_k,d|^p` and `eps = r * S_typ` (relative epsilon; `S_typ` is the mean
  of `S` over allowed pairs at initialisation, measured once and frozen). `p = inf` uses `max_d |.|` (no epsilon), with the
  gradient split evenly among tied coordinates.
* Distances are computed in float32. Special-case powers avoid the generic `pow` (p = 1, 1.5, 3, 4, 8).
  `p = 2` uses the matmul expansion `|a|^2 + |b|^2 - 2 a.b` clamped at zero (a run with the direct-difference kernel gave the same accuracy).
* The kernel (`msa/lp.py`) is a custom autograd function: the forward is chunked over queries and saves only `q`, `k` and the output;
  the backward recomputes differences chunk by chunk with the analytic gradient `dist^(1-p) |Delta|^(p-1) sign(Delta)`.
  For the language model the chunks skip keys beyond the causal horizon. Default chunk: 4 (fastest in the sweep).

## Calibration (msa/calibration.py)

* At initialisation, on a model built with a fixed calibration seed and fixed calibration batches (identical for all arms and seeds):
  for each layer in order, compute for every query the standard deviation of the raw score over its allowed keys, average over
  queries (with at least 8 allowed keys), heads and batches, and set `c = target / std_raw`. The target spread is 1.
* The diagonal is masked in the language model (position 0 may attend to itself and is excluded from the spread because it has one key).
  In the recall task all `P` context keys are allowed and no diagonal exists.
* One constant per (task configuration, arm, layer); the same constants are used for every seed.

## Models

* Recall (MQAR variant A): key table (256 x 64), value table (65 x 64, last row = blank value), token = key embedding + value embedding,
  LayerNorm, one head of dimension D (tied projection), value projection and output projection of full width, residual from the
  query embedding, LayerNorm, linear readout to 64 value classes; no MLP, no positions. Queries attend only to the P context tokens.
  Q = P/2 queries per sequence, loss at query positions only.
* Character LM: 2 pre-norm decoder layers, d_model 64, 4 heads of dimension 16, MLP 4x with GELU, learned absolute positions,
  context 64, tied input/output embeddings, no biases in attention, no dropout, no QK-normalisation. About 99.7k parameters (107.8k untied).
* Initialisation: each parameter is drawn from a generator seeded by (run seed, parameter name), so a parameter that exists in two arms has
  identical initial values in both. Linear weights ~ N(0, 1/fan_in) (residual output projections scaled by 1/sqrt(2 layers)),
  embeddings N(0, 0.02) in the LM and N(0, 1) in the recall task.

## Training protocol (defaults in msa/train.py; the pilot results say which values were adopted)

* AdamW (betas 0.9, 0.99), weight decay 0.01 on matrices (not on embeddings, norms, biases, beta), warm-up then cosine decay to 10% of the peak
  rate, gradient clipping at norm 1.0 (the pre-clip norm is logged).
* Batches are generated from a seeded stream (data seed = 1000 + run seed), so arms with the same seed see identical data.
  Validation and test sets for the recall task are fixed (5000 sequences each, separate seeds, identical for every arm).
  The language-model validation set is the last 10% of the text, evaluated on non-overlapping windows of 64 characters.
* Diagnostics (every `diag_every` steps and at the end): per head beta, max |logit|, logit spread, attention entropy, q and k norms, and the
  gradient statistics of the projected vectors. Recall-task final diagnostics also give top-1 attention accuracy, attention mass on the
  correct key, logit and distance margins, and the hub measure (correlation between key norm and attention received).
* Checkpoints every 250 steps; resuming is bit-exact (tested). `final.pt` (weights, config, constants) is saved for every run.
