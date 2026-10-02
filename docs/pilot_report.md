# Pilot report (2026-10-02)

All numbers come from `results/pilot/*.json`. The pilot ran in a Linux cloud sandbox (2 vCPU Xeon, torch 2.2.2, Python 3.11).
Timings were also measured in the Linux VM that your desktop app exposes (4 vCPU, i5-1038NG7, torch 2.2.2, Python 3.10; `results/pilot_vm/`).
Neither machine is macOS, so every timing is a proxy. Please re-run `python execute_project.py bench` natively.
Runs from the early stages used r = 1e-6 and the matmul path for p = 2. Later stages use r = 1e-12 and the direct kernel (see section 5 and 6).

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
