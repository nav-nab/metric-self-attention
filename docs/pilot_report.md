# Pilot report (2026-10-02)

> **Note (2026-10-05):** this report is a snapshot of the pilot as it was reported. The recall task (MQAR) has since been shelved, and its code, design record and findings are in `shelved/mqar/`. The report still refers to the combined entry point of that time (stages such as `transition` and `generator` now live in `shelved/mqar/execute_mqar.py`). The numbers in `results/pilot/` are unchanged.

All numbers come from `results/pilot/*.json`. The pilot ran in a Linux cloud sandbox (2 vCPU Xeon, torch 2.2.2, Python 3.11).
Timings were also measured in the Linux VM that your desktop app exposes (4 vCPU, i5-1038NG7, torch 2.2.2, Python 3.10; `results/pilot_vm/`).
Neither machine is macOS, so every timing is a proxy. Please re-run `python execute_project.py bench` natively.
Runs from the early stages used r = 1e-6 and the matmul path for p = 2. Later stages use r = 1e-12 and the direct kernel (see section 5 and 6).

## Glossary of symbols and terms
Every symbol and abbreviation used in this report is defined here. The two tasks are the recall task (MQAR) and the language model (LM).

### Experimental design
* **Arm**: one score function being compared. There are nine: `dot_tied`, `lp_1`, `lp_1.5`, `lp_2`, `lp_3`, `lp_4`, `lp_8`, `lp_inf`, and `dot_untied`. The name `lp_p` means the negative L^p distance with that p. In the text L1, L2, L8 and L-inf (or L-infinity) are short for `lp_1`, `lp_2`, `lp_8` and `lp_inf`.
* **Tied / untied**: tied means queries and keys use the same projection, W_Q = W_K. Untied means separate W_Q and W_K, as in a standard transformer. Only `dot_untied` is untied.
* **dot**: the dot-product score. "dot" alone means `dot_tied` unless the text says untied.
* **L^p arms**: the seven distance arms (p = 1, 1.5, 2, 3, 4, 8, infinity).
* **Cell**: one (P, D) combination of the recall task, for example (P=64, D=2). Cells are written (P, D).
* **Run**: one training of one arm in one cell (or one LM configuration) with one seed.
* **Seed (s0, s100, ...)**: the number that fixes initialisation and the data stream of a run. Seeds are paired: every arm with the same seed shares the same data and the same initial values for the parameters it has in common.
* **Paired difference**: for one seed, the metric of one arm minus the metric of another arm, for example L2 minus dot. The report gives the mean and the standard deviation (sd) of this difference over seeds.
* **sd**: standard deviation across seeds.
* **Minimal detectable difference (MDE)**: the smallest true difference a paired t-test would detect with 80% power. It is quoted in units of the sd of the paired difference. "Uncorrected" means a single test at the 5% level. "Seven-comparison correction" means the Holm correction over the seven distance arms.
* **Transition region**: the range of P (at fixed D) where accuracy falls from near 1 to near chance.

### Recall task (MQAR variant A)
* **P**: number of key-value pairs in the context of one sequence. It is the difficulty knob.
* **Q**: number of queries per sequence. Always Q = P/2.
* **Sequence length**: P + Q positions. The first P positions hold the context pairs. The last Q positions are queries.
* **Context pair [k, v]**: one context token. Its embedding is the key embedding plus the value embedding.
* **Query [k, _]**: a query token. It has the same key embedding as in the context and a special blank value embedding. The model must output the value that was paired with k.
* **Keys and values**: 256 distinct key symbols and 64 distinct value symbols. Keys in one sequence are distinct. Values are drawn with replacement. The output is a classification over the 64 values.
* **Chance**: 1/64 = 0.0156, the accuracy of a model that cannot see the context.
* **Key-blind baseline**: the accuracy of guessing a random value from the context, 1/P + (1 - 1/P)/64.
* **1024 keys**: the extended vocabulary used only for the probes at P = 128 and 256. Keys must stay distinct inside a sequence, so 256 keys is too few there.
* **Query-to-context mask**: each query attends only to the P context tokens. It does not attend to other queries or to itself.
* **Accuracy**: the fraction of queries answered correctly on the fixed test set (5000 sequences, identical for all arms).
* **Top-1 attention**: the fraction of queries whose largest attention weight falls on the correct context pair.
* **Steps to 99%**: the first evaluation step (every 100 steps) at which validation accuracy reaches 0.99. Steps to 50% and 90% are defined the same way.
* **Hub measure**: the correlation across context tokens between key norm and the attention they receive. A positive value means long keys act as hubs.
* **Shuffle check / no-context check**: sanity tests. Shuffling the order of the context pairs must not change accuracy. Removing the context must give chance.

### Models
* **d_model**: width of the residual stream and of the embeddings. Both tasks use 64.
* **H**: number of attention heads. The recall task has H = 1. The LM has H = 4.
* **D**: head dimension, the dimension of the space where queries and keys are compared. The recall task sweeps D in {1, 2, 3, 4, 6, 8, 16}. The LM uses D = 16 (4 heads of 16).
* **z, z_q, z_k**: the query and key vectors after the projection (D numbers each). With tied projections z_q and z_k come from the same matrix.
* **Layer (1 / 2)**: the LM has two attention layers. The recall model has one.
* **MLP**: the feed-forward block in a transformer layer. The recall model has none. The LM has one per layer, with width 4 x d_model.
* **Learned absolute positions**: one trained vector per position added to the embedding. Only the LM uses them.
* **Diagonal masked**: a position may not attend to itself (position 0 may, so it has something to attend to). This applies to the LM.
* **Causal mask**: a position may attend only to earlier positions.
* **Logit**: the raw attention score before the softmax.
* **Max |logit|**: the largest absolute logit seen in a batch. Large values signal unstable attention.
* **Attention entropy**: the entropy of the softmax weights, in nats. A value of 0 means all weight sits on one key. The maximum is the log of the number of allowed keys.
* **q norm**: the length of the query vector. Growth of this norm is one symptom of instability.

### Score function and its constants
* **p**: the exponent of the L^p distance. The grid is 1, 1.5, 2, 3, 4, 8 and infinity.
* **Distance score**: logit = -beta x c x dist, where dist = (S + eps)^(1/p).
* **S**: the sum inside the root, S = sum over the D coordinates of |z_q - z_k|^p. For p = infinity, dist is the largest coordinate difference instead.
* **eps**: a small number added inside the root so the gradient is finite at distance zero.
* **r**: the relative size of eps. eps = r x S_typ. The proposed value is r = 1e-12.
* **S_typ**: the typical value of S, measured on real batches at initialisation and then frozen. It is given per arm and per layer.
* **Dot score**: logit = beta x c x (z_q . z_k).
* **beta**: a learnable scale per head, beta = exp(log_beta), initial value 1.
* **c**: a frozen calibration constant per arm, layer and configuration. It is chosen so that the standard deviation of the logit across allowed keys equals the target spread at initialisation.
* **Target spread**: the standard deviation of the logits that calibration aims for. It is provisionally 1.
* **Calibration**: the measurement of c and S_typ on fixed batches with a calibration-seed model.
* **1/sqrt(D)**: the standard dot-product scaling. The calibrated dot constant should land near it. For D = 16 it is 0.25.
* **Matched pair**: a query and the context key that holds its answer.

### Training
* **lr**: learning rate, the peak step size of the optimiser. Written as 1e-3 = 0.001, 3e-2 = 0.03.
* **Schedule**: 100 warm-up steps with a rising rate, then cosine decay to 10% of the peak.
* **Step**: one optimiser update on one batch.
* **B**: batch size. The LM uses B = 64 sequences. The recall task uses B = 128.
* **T**: LM context length, the number of characters per training sequence. T = 64.
* **Steps per run**: 2500 for the LM (about 10M characters seen) and 3000 for recall.
* **AdamW, weight decay (wd), clipping**: AdamW is the optimiser with betas (0.9, 0.99). Weight decay is 0.01 on matrices. Clipping rescales the gradient when its norm exceeds 1.0.
* **Pre-clip norm**: the gradient norm before clipping. If it often exceeds 1.0, clipping is active.
* **Divergence**: a loss that becomes non-finite or explodes. None occurred in the pilot.
* **Chunk**: the number of queries processed at a time by the custom distance kernel. It trades speed for memory. The default is 4.
* **Causal block skipping**: the kernel skips key blocks that no query in the chunk may attend to. It saves 30 to 50% in the LM.
* **Custom / checkpointed / naive**: three implementations of the distance. Naive uses plain autograd and is the reference. Checkpointed recomputes pieces in the backward pass. Custom is the hand-written autograd function with an analytic backward.
* **Matmul path (p = 2)**: computing the L2 distance from |a|^2 + |b|^2 - 2 a.b. It is fast but loses accuracy at small D. The direct path computes the differences explicitly.
* **Peak RSS**: the largest resident memory of the process, in MB.
* **Workers**: parallel training processes, each on one thread.

### Metrics
* **bpc**: bits per character, the LM cross-entropy in bits. Lower is better. Validation bpc is measured on fixed windows of the held-out 10% of the text. A uniform guess over 65 characters would give 6.02 bpc.
* **Tiny Shakespeare**: the LM corpus. It has 1,115,394 characters and 65 distinct symbols.
* **Test accuracy (recall)**: see Accuracy above.

### Files
* `results/pilot/*.json`: summaries of each pilot stage.
* `results/pilot/<run id>/`: one directory per run, with `result.json` and `final.pt` (final weights). A run id looks like `mqar_P64_D3_dot_tied_s0`: task, P, D, arm, seed.
* `execute_project.py`: the single entry point that runs every stage.

## 1. Environment
torch 2.2.2 and numpy 1.26 work in both environments, and a smoke training step runs for both tasks and the dot, L^3 and L-infinity arms.
Python must be 3.10 to 3.12. `requirements.txt` pins torch 2.2.2 and numpy<2.
The native macOS Intel install could not be tested from here.

## 2. Gradient check (float64)
The custom kernel, the checkpointed kernel and the p = 2 matmul path match naive autograd for p in {1, 1.5, 2, 3, 4, 8, inf}, with and without causal block skipping.
The worst absolute error over outputs, dq and dk is 7e-15. `torch.autograd.gradcheck` passes for all p.
Chebyshev: custom and naive amax agree on tied data, and four tied coordinates each get 1/4 of the gradient. The float32 kernel is within 3.4e-7 relative error of float64.

## 3. Generator check
For P in {8, 16, 32, 64} (Q = P/2, 256 keys, 64 values): keys distinct, queries drawn from the context, each key queried once, answers consistent, uniform value, key and queried-position frequencies (chi-square), and both baselines match theory
(for example P = 16: chance 0.0161 vs 0.0156, key-blind 0.0771 vs 0.0771). The vectorised generator and the reference generator agree on all checks. Fixed validation and test sets are reproducible.
Sanity checks hold for every trained run: shuffling the context leaves accuracy unchanged (difference 0), and removing the context gives chance.

## 4. Timing and memory (training step, fp32; sandbox with 2 threads / laptop VM with 4 threads; peak RSS)
Language model (B = 64, T = 64): dot 0.057 s / 344 MB; L1 0.157 s; L1.5 0.27 s; L2 0.147 s (direct kernel; 0.02 s with matmul); L3 0.21 s; L4 0.19 s; L8 0.235 s; L-inf 0.27 s / 478 MB; untied dot 0.06 s.
The VM gives 0.038 s (dot), 0.15 s (L3), 0.154 s (L-inf) with 4 threads. Peak memory is 340 to 640 MB for every arm.
Recall (B = 128): P=64, D=16: dot 0.022 s, L^p 0.03 to 0.05 s; smaller cells 0.008 to 0.02 s.
Kernel implementations at the LM size: naive 0.65 s / 921 MB, checkpointed 0.55 s / 637 MB, custom 0.215 s / 431 MB (L3). Chunk size: 2 and 4 are fastest; 16 is 1.6x slower; 32 is 2.4x slower.
Default chunk is 4. Causal block skipping saves 30 to 50%.
Throughput: two single-thread processes in parallel give 1.13x (sandbox) and 1.38x (VM, four processes) the throughput of one process using all threads. Use `--workers`.
Measured full-run cost on the sandbox (single thread, two workers active): LM 2500 steps takes 234 s (dot), 443 s (L2), 819 s (L-inf).
Recall runs (3000 steps) take 50 to 190 s at P = 32 to 64.

## 5. Calibration, typical sums, epsilon
Target logit spread 1 (provisional, not yet chosen by you). Constants c per layer (LM): dot 0.225 / 0.242 (1/sqrt(D) = 0.25, so the sanity check holds);
L1 0.306 / 0.327; L1.5 0.72 / 0.75; L2 1.05 / 1.09; L3 1.43 / 1.50; L4 1.58 / 1.68; L8 1.65 / 1.82; L-inf 1.60 / 1.82; untied dot 0.27 / 0.34.
Typical sum inside the root S_typ (layer 1 / layer 2): L1 17.5 / 13.4; L1.5 22.1 / 15.1; L2 30.1 / 18.5; L3 66.6 / 32.7; L4 174 / 68; L8 2.3e4 / 4.3e3 (mean far above the median 5.0e3 / 5.3e2).
Recall constants are in `results/pilot/calibrate.json` for D in {2, 4, 8, 16}. For D = 16 the dot constant is 0.231 (P = 16) and 0.241 (P = 64).
Proposed r = 1e-12. Evidence:
* r = 0 produces non-finite gradients for every p > 1 at exact duplicates. Any r > 0 is finite in float32.
* Trained matched pairs reach sums far below S_typ: for L8 the 1st percentile of the matched-pair sum is 1e-14 of S_typ. The floor r^(1/p) therefore matters for large p.
* Accuracy at (P = 64, D = 3), two seeds, lr 1e-3: L8 gives 0.992 / 0.998 (r = 1e-12), 0.983 / 0.982 (1e-6), 0.950 / 0.947 (1e-3). L4 and L2 are unaffected below 1e-6.
* The p = 2 matmul path is accurate to the same degree as the direct kernel at D >= 2 (0.9967 vs 0.9967), but at D = 1 it loses accuracy (0.76 vs 0.99). I therefore switched L2 to the direct kernel. This deviates from pilot-plan item 2 and costs 3x the L2 step time.

## 6. Learning-rate safeguards
First test (lr 1e-3 and 3e-3, 400 steps on the full-length schedule, all nine arms, both tasks): no divergence.
Second test (LM 3e-3 and 9e-3, recall 3e-2 and 9e-2, recall cell P = 64, D = 2): no divergence for any arm.
At 3e-3 and 9e-3 the LM max |logit| stays at 17 to 39 and q norms grow by at most 10%. On recall at 3e-2 and 9e-2 max |logit| is 53 to 340, and q norms grow 3x to 7x (all arms). Attention entropy falls to 0.5 to 0.8 nats (L^p) against 2.3 (dot).
Gradient clipping at 1.0 is active in the early phase for all arms (log windows with a pre-clip norm above 1: 12 to 75%). Untied dot is clipped most often.
Full-length LM runs (2500 steps) at three rates, validation bpc: dot 2.545 / 2.396 / 2.394, L2 2.551 / 2.430 / 2.405, L-inf 2.577 / 2.432 / 2.438 at lr 1e-3 / 3e-3 / 1e-2. No divergence. 1e-3 under-trains every arm.
Proposed LM rate 3e-3 (flat between 3e-3 and 1e-2). The 3x trial passes.

Important finding for the recall task: the tied dot arm needs a much higher rate than the L^p arms.
Test accuracy at lr 1e-3 / 3e-3 / 1e-2 / 3e-2 / 1e-1 / 3e-1:
* (P=64, D=3): dot_tied 0.513 / 0.678 / 0.690 / 0.997 / 0.999 / 1.000; untied dot 0.528 / 0.814 / 0.979 / 0.997 / 0.546 / 0.025; every L^p arm 0.98 to 1.000 from 1e-3 on.
* (P=32, D=2): dot_tied 0.327 / 0.378 / 0.555 / 0.930 / 0.963 / 0.979; untied dot 0.345 / 0.445 / 0.746 / 0.907 / 0.974 / 0.978; L^p arms 0.90 to 0.96 at 1e-3 and 1.000 from 1e-2.
* Long training (12000 steps) at (64, 3): dot_tied 0.692 (lr 1e-3), 0.806 (3e-3) against 0.513 and 0.678 at 3000 steps.
Reading: the gap seen at lr 1e-3 is mostly optimisation speed. A shared rate is either unfair to the dot arms (1e-3 to 1e-2) or close to the stability edge of the untied arm (above 3e-2).

## 7. Seed variance (5 paired seeds, seeds 100 to 104)
Recall at lr 3e-2, 3000 steps:
* (P=64, D=2): dot_tied 0.850 (sd 0.0069), untied dot 0.856 (sd 0.0059), all seven L^p arms 1.0000 (sd 0). Paired difference L^p minus dot_tied: +0.150, sd 0.0069. Untied minus tied: +0.0057, sd 0.0087.
* (P=32, D=2): dot_tied 0.926 (sd 0.012), untied 0.915 (sd 0.0036), L^p arms 1.0000. Paired difference +0.074, sd 0.012. Untied minus tied: -0.0115, sd 0.0135.
Language model at lr 3e-3, 2500 steps, validation bpc (5 seeds): dot_tied 2.4107 (sd 0.0096), L2 2.4060 (sd 0.0101), L-inf 2.4271 (sd 0.0102).
Paired differences: L2 minus dot -0.0046 (sd of the difference 0.0163, correlation -0.36); L-inf minus dot +0.0165 (sd 0.0100, correlation 0.48).
Pairing does not reduce variance on the LM (correlations are small and of both signs with n = 5). Use the sd of the difference, 0.010 to 0.016 bpc.
With 5 seeds the minimal detectable difference (80% power) is 1.69 sd uncorrected (0.017 to 0.028 bpc) and 2.88 sd with the seven-comparison correction (0.029 to 0.047 bpc).
The L^p arms are at the ceiling in both recall cells, so the recall seed variance of an L^p arm and the differences among p cannot be estimated there.

## 8. Recall transition region
At lr 1e-3 (1 seed, P in {8..64}, D in {2..8}) dot_tied crosses 90% near P = 22 at D = 4 and is below 90% at every P for D <= 3. L^p arms cross near P = 34 to 37 at D = 2 and stay above 90% for D >= 3.
At lr 3e-2 (1 seed, D in {1, 2, 3, 4}):
* D = 4: every arm is 1.000 at all P.
* D = 3: every arm is at least 0.995 for P <= 64.
* D = 2: dot_tied 0.93 / 0.95 / 0.93 / 0.43 at P = 8 / 16 / 32 / 64 (untied 0.94 / 0.94 / 0.91 / 0.86); L1, L2, L-inf are 1.000 everywhere.
* D = 1: all L^p arms are the same function in one dimension. L1 and L-inf give 0.98 / 0.99 / 0.83 / 0.70, dot 0.64 / 0.46 / 0.29 / 0.18; L2 with the old matmul path gave 0.94 / 0.76 / 0.30 / 0.17 (a numerical artefact, see section 5).
Probe beyond the guide (1024 keys, lr 3e-2, 1 seed): D = 3, dot_tied 0.985 at P = 128 and 0.336 at P = 256, L2 1.000 at both. D = 2, dot_tied 0.330 at P = 128 and 0.279 at P = 256, L2 1.000 and 0.986.
Run times grow quickly with P (L2, single thread under load): 65 s at P = 32, 170 s at P = 64, 580 s at P = 128, 2600 s at P = 256.
Consequence: with P <= 64 and an adequate learning rate, the only informative cell for dot against distance is D = 2. The L^p arms are saturated, so their differences cannot be measured by final accuracy.
The primary outcome as written (the P where accuracy falls below 90% at a fixed D) cannot be computed for all arms in the same D within P <= 64.
Hub measure: dot arms correlate key norm with attention received (0.57 at P=64, D=3, lr 1e-3); L^p arms are near 0 (-0.02 to 0.0).
Per-cell training is cheap (under 3 minutes per run at P <= 64), so a mixture over P is not needed.

## 9. Budget (sandbox seconds; the laptop VM was about 1.4x faster per thread)
Per-run cost (two workers active): recall 20 s (P = 8) to 190 s (P = 64); LM 206 s (untied dot), 234 s (dot), 443 s (L2), 536 s (L1), 650 to 920 s (other L^p), 819 s (L-inf).
Projection for a 16-cell recall grid (D in {2,3,4,6} x P in {8,16,32,64}), 9 arms, 5 seeds, plus the LM with 9 arms: 3 seeds 8.3 h, 4 seeds 9.1 h, 5 seeds 9.8 h of wall time on two workers.
This is far below the 48 h. Spare time covers the optional extras and an extended-P recall grid. The pilot itself used about 20 compute hours in the sandbox.
