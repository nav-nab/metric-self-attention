# MQAR design record

This file collects the design decisions for the recall task. Decisions marked **[user]** were stated by the user and are the authority.
Items marked **[Claude]** were implementation choices made by Claude and not confirmed. Decisions that the recall task inherits from the
main project are listed separately, so that a reviver can check them against the current specification.

## Task and data

- **[user]** The task is variant A of multi-query associative recall (MQAR; Arora et al., Zoology). The answers come from a deterministic
  rule, so the model must learn the lookup.
- **[user]** A context token is the key embedding plus the value embedding. A query token is the same key embedding plus a special blank
  value embedding. The output is a classification over the values, with the loss at the query positions only.
- **[user]** Keys within a sequence are distinct (drawn without replacement). Values are drawn with replacement. Each key is queried at
  most once. The order of the context pairs is irrelevant, so there are no positional embeddings.
- **[user]** Training batches are generated fresh from a seeded generator. The validation and test sets are fixed (5000 sequences each),
  come from separate seeds, and are identical for every arm.
- **[user]** The number of queries is Q = P / 2, to weaken an elimination shortcut. There are 256 keys and 64 values for P up to 64.
- **[user]** The baselines to report are the no-context chance, 1/V, and the key-blind guess of a random context value, 1/P + (1 - 1/P)/V.
  Retrieval must clearly beat the key-blind baseline.
- **[user]** Sanity checks: shuffling the context must leave accuracy unchanged, and a model that cannot see the context must score at chance.

## Model

- **[user]** One attention layer, one head, no MLP, d_model = 64. The head dimension D is the whole query-key space.
- **[user]** Each query attends only to the P context tokens, not to earlier queries. There is no diagonal to mask, so the masked-diagonal
  rule of the language model does not apply here.
- **[user]** The write-up must say that the recall task uses this query-to-context pattern, and that multi-head evidence comes only from
  the language model.
- **[Claude]** Layer norm on the input, value width equal to d_model, output projection W_o, layer norm before a linear readout. Embeddings
  are initialised N(0, 1) (the language model uses N(0, 0.02^2)).

## Measurements

- **[user]** Accuracy against P and D, the smallest D that reaches near-perfect accuracy, steps to reach a set accuracy, the hub measure
  (correlation of key norm with attention received), distance margins, attention entropy and top-1 accuracy.
- **[user, to revisit]** Whether to train one model per (P, D) cell or one model on a mixture over P.
- **[user]** The primary outcome was to be the pair count P at which accuracy falls below 90% at a fixed D. The pilot showed that this
  cannot be computed for the L^p arms (see `PILOT_FINDINGS.md`), so it needs to be redefined.

## Decisions shared with the main project

These apply to the recall task as they apply to the language model. They are repeated here so that they can be checked at revival time.

- **[user]** Tied projections (W_Q = W_K) for every arm except the untied dot-product reference.
- **[user]** The score is the negative L^p distance, scaled by a learned per-head factor beta (initial value 1) and a frozen calibrated
  constant c. For p = infinity the distance is the largest coordinate difference.
- **[user]** The constant c is calibrated at initialisation so that the standard deviation of the scores across the allowed keys equals 1.
  It is recomputed for every head size, arm and layer, on a calibration-seed model with fixed batches. Matching at initialisation is all that
  is done, and the scale is not tuned per arm.
- **[user]** The relative epsilon: eps = r * (typical sum inside the root). The value r = 1e-12 is proposed and **not yet approved**.
- **[Claude, not approved]** p = 2 uses the direct difference kernel, not the matmul expansion.
- **[user]** No normalisation of queries and keys. Final weights are saved from every run. All arms use the same data and the same
  starting values of shared parameters for a given seed.

## Implementation choices (Claude)

- AdamW (0.9, 0.99), weight decay 0.01 on matrices, 100 warm-up steps and cosine decay to 10%, gradient clip 1.0, 3000 steps, batch 128.
- The default learning rate in `task.py` is the common default 1e-3. The pilot found that this under-trains the dot arms and that 3e-2 is
  much better for them.
- The pilot cell (the (P, D) cell where the mean accuracy of tied dot and L2 is closest to 0.75) is chosen by the transition stage.

## Open points to settle before a revival

1. The learning rate for each arm, or a protocol that reports results across rates.
2. A redefined primary outcome, since the original cannot be computed.
3. The value of r, and the p = 2 kernel.
4. Whether to use a task variant with a harder matching rule, so that the values of p can be ranked.
5. Whether to share a backbone with the language model (the user rejected this proposal on 2026-10-05).
