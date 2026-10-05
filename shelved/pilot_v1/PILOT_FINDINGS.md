# Version 1 pilot: findings for the language model and the shared machinery

The full report is `docs/pilot_report.md`, with a glossary of every term. The raw outputs are in `results/pilot/`. The recall findings
are in `mqar/PILOT_FINDINGS.md`. This file lists the findings that concern the language model and the shared code. For each finding it
says whether it bears on the restarted project.

## Caveats that apply to every number

- Every pilot run used head size 16 only. The restarted project varies the head size, so no pilot number describes a small head.
- The numbers come from a cloud sandbox (2 vCPU) and the Linux VM of the desktop app (4 vCPU). Neither is native macOS.
- The early stages used r = 1e-6 and the matrix-multiply path for p = 2. The later stages used r = 1e-12 and the direct kernel.
- The language-model numbers are for one learning rate (3e-3) with 5 seeds, and for a short scan at three rates with one seed.

## Findings

**1. The distance kernel is correct.** The custom kernel, the checkpointed kernel and the matrix-multiply path match naive autograd in
float64 for every p, with and without causal block skipping. The worst absolute error is 7e-15. Chebyshev ties split the gradient evenly.
The float32 kernel is within 3.4e-7 relative error of float64. *Bears on the restart: yes, the kernel in `msa/lp.py` could be reused.*

**2. Step times and memory (language model, batch 64, context 64, sandbox, 2 threads).** Dot product 0.057 s; L1 0.157 s; L1.5 0.27 s;
L2 0.147 s (direct kernel); L3 0.21 s; L4 0.19 s; L8 0.235 s; L-inf 0.27 s; untied dot 0.06 s. The custom kernel is about 3 times faster
and uses about half the memory of naive autograd. Chunk size 2 to 4 is fastest. Causal block skipping saves 30 to 50%. Peak memory is 340
to 640 MB per process. Two single-thread processes give 1.13 times (sandbox) and four give 1.38 times (VM) the throughput of one process
with all threads. *Bears on the restart: yes for the budget, but the timings must be re-measured, and the cost at small head sizes was
only estimated.*

**3. Full-run costs (2500 steps, one thread, two workers active, sandbox).** Dot 234 s, untied 206 s, L1 536 s, L1.5 918 s, L2 443 s,
L3 720 s, L4 650 s, L8 798 s, L-inf 819 s. *Bears on the restart: yes, as the basis of the budget estimate for the head-size grid.*

**4. Calibration constants (language model, D = 16, target spread 1).** The tied dot constant is 0.225 and 0.242 in the two layers, against
1/sqrt(16) = 0.25, so the sanity check holds. The distance constants grow with p, from 0.31 (L1) to 1.8 (L8 and L-inf in layer 2). The
typical sum inside the root for L8 has a mean far above its median, so the distribution is heavily skewed. *Bears on the restart: yes,
the sanity check against 1/sqrt(D) carries over, and the skew supports the caveat that one standard deviation is a crude summary.*

**5. Epsilon.** With r = 0 the gradient is non-finite at exact duplicates for every p > 1. Any r > 0 is finite in float32. A relative r of
1e-12 was proposed. On the recall task, accuracy for p = 8 fell for r of 1e-6 and above. *Bears on the restart: partly. The
gradient-safety result carries over. The accuracy evidence comes from the recall task and is not directly transferable.*

**6. Learning rate (language model, full length, one seed).** Validation bpc at 1e-3, 3e-3 and 1e-2: dot 2.545, 2.396, 2.394; L2 2.551,
2.430, 2.405; L-inf 2.577, 2.432, 2.438. A rate of 1e-3 under-trains every arm. The curve is flat between 3e-3 and 1e-2. Nothing
diverged at 3 times the base rate. The maximum logit stays at 17 to 39 and the query norms grow by at most 10%. Gradient clipping is
active early for all arms. *Bears on the restart: yes. The rate matters, and the pilot's observation that the dot arm needed a far higher
rate at small head size (on the recall task) is why the restart plans a rate scan at each D.*

**7. Seed variance (language model, learning rate 3e-3, 5 paired seeds).** Validation bpc: dot 2.4107 (sd 0.0096), L2 2.4060 (sd 0.0101),
L-inf 2.4271 (sd 0.0102). The paired difference of L2 minus dot is -0.0046 (sd 0.0163), and of L-inf minus dot is +0.0165 (sd 0.0100).
Pairing did not reduce the variance. With 5 seeds the smallest detectable difference is about 0.017 to 0.028 bpc uncorrected and 0.029
to 0.047 with the seven-comparison correction. *Bears on the restart: yes. This is the noise floor behind the statement that only
differences above about 0.02 bpc are detectable.*

**8. The arms are indistinguishable at head size 16.** At D = 16 the tied dot product, L2 and L-inf are within seed noise, with L-inf
slightly worse. Only three of the nine arms were trained to full length. *Bears on the restart: yes, it is the reason for the small-head
design.*

**9. The language model underfits.** Training loss was about 2.13 bpc against 2.41 on validation, and a model of this kind could reach
about 1.5 to 1.7. This is a capacity-limited regime. *Bears on the restart: yes, it is a stated limitation.*

**10. Environment.** torch 2.2.2 with numpy below 2 works on Python 3.10 to 3.12 in both Linux environments. The native macOS Intel
install was not tested.

## What the pilot did not measure

Any small head size; the uniform-attention control; the learning-rate scan per head size; the full nine-arm language-model grid with
several seeds; native macOS timings.
