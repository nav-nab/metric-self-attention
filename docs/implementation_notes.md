# Implementation notes (for the write-up)

These notes describe what the code in this repository actually does. Design decisions are in the project guide
(kept with the research notes). Numbers from the pilot are in `results/pilot/` and summarised in `docs/pilot_report.md`.

## Arms and the score

* Nine arms: `dot_tied`, `lp_1`, `lp_1.5`, `lp_2`, `lp_3`, `lp_4`, `lp_8`, `lp_inf`, `dot_untied`, plus the control arm `uniform`
  (constant score, so every allowed key gets the same weight; it has no scale to calibrate).
* Every arm uses one score module (`msa/models.py: Scorer`).
  Distance arms: `logit = -beta_h * c * dist_p(z_q, z_k)`; dot arms: `logit = beta_h * c * (z_q . z_k)`.
  `beta_h = exp(log_beta_h)` is learnable per head and starts at 1 in every arm. `c` is a frozen calibrated constant
  (a buffer, saved with the weights). Tied arms use `z = W u` for queries and keys; `dot_untied` has separate `W_q`, `W_k`.
* Distance: `dist = (S + eps)^(1/p)` with `S = sum_d |z_q,d - z_k,d|^p` and `eps = r * S_typ` (relative epsilon; `S_typ` is the mean
  of `S` over allowed pairs at initialisation, measured once and frozen). `p = inf` uses `max_d |.|` (no epsilon), with the
  gradient split evenly among tied coordinates.
* Distances are computed in float32. Special-case powers avoid the generic `pow` (p = 1, 1.5, 3, 4, 8).
  `p = 2` uses direct differences like every other p (`impl="auto"`). The matmul expansion `|a|^2 + |b|^2 - 2 a.b` clamped at zero is opt-in
  (`impl="matmul"`), because it loses accuracy when distances are tiny. Use of the direct kernel is a Claude choice awaiting approval.
* The kernel (`msa/lp.py`) is a custom autograd function: the forward is chunked over queries and saves only `q`, `k` and the output;
  the backward recomputes differences chunk by chunk with the analytic gradient `dist^(1-p) |Delta|^(p-1) sign(Delta)`.
  For the language model the chunks skip keys beyond the causal horizon. Default chunk: 4 (fastest in the sweep).

## Calibration (msa/calibration.py)

* At initialisation, on a model built with a fixed calibration seed and fixed calibration batches (identical for all arms and seeds):
  for each layer in order, compute for every query the standard deviation of the raw score over its allowed keys, average over
  queries (with at least 8 allowed keys), heads and batches, and set `c = target / std_raw`. The target spread is 1.
* The diagonal is masked in the language model (position 0 may attend to itself and is excluded from the spread because it has one key).
* One constant per (task configuration including head size, arm, layer); the same constants are used for every seed.

## Models

* Character LM: 2 pre-norm decoder layers, d_model 64, 4 heads, MLP 4x with GELU, learned absolute positions,
  context 64, tied input/output embeddings, no biases in attention, no dropout, no QK-normalisation. About 99.7k parameters (107.8k untied).
  The causal mask has the diagonal masked, and position 0 may attend to itself.
* Head size: `qk_dim` (config field, default `d_model / n_heads` = 16) sets the query-key width of each head. `W_q` and `W_k` are
  `Linear(d_model, n_heads * qk_dim)`. The value width per head stays `d_model / n_heads`, so the rest of the model is unchanged.
  Calibration constants, run ids (`lm_D<qk_dim>`) and cell names include `qk_dim`, so each head size is calibrated separately.
* Initialisation: each parameter is drawn from a generator seeded by (run seed, parameter name), so a parameter that exists in two arms has
  identical initial values in both. Linear weights ~ N(0, 1/fan_in) (residual output projections scaled by 1/sqrt(2 layers)),
  embeddings N(0, 0.02). Parameters whose shape depends on `qk_dim` differ between head sizes, and all other parameters are identical across head sizes.

## Task plug-ins (msa/train.py)

The engine does not know any task. A task subclasses `Task` (defaults, calibration fields, model builder, calibration batches, data setup,
training loss, evaluation, result header, final metrics) and is registered by name. The language model is the only built-in task.
A config selects a task by name. A config of an external task also carries `task_module`, so that a worker started with `spawn` imports
the task first. The shelved recall task is such a plug-in (`shelved/mqar/task.py`). When the engine was split from the old combined
version, language-model and recall runs were compared against the old engine, and weights, results and run ids were bit-identical.

## Training protocol (defaults in msa/train.py; the pilot results say which values were adopted)

* AdamW (betas 0.9, 0.99), weight decay 0.01 on matrices (not on embeddings, norms, biases, beta), warm-up then cosine decay to 10% of the peak
  rate, gradient clipping at norm 1.0 (the pre-clip norm is logged).
* Batches are generated from a seeded stream (data seed = 1000 + run seed), so arms with the same seed see identical data.
  The language-model validation set is the last 10% of the text, evaluated on non-overlapping windows of 64 characters.
* Diagnostics (every `diag_every` steps and at the end): per head beta, max |logit|, logit spread, attention entropy, q and k norms, and the
  gradient statistics of the projected vectors.
* Checkpoints every 250 steps; resuming is bit-exact (tested). `final.pt` (weights, config, constants) is saved for every run.

The recall task (model, generator, evaluation, pilot stages) is documented in `shelved/mqar/`.
