# MQAR pilot: findings and lessons

These are the results from the original pilot (one seed unless stated), written down so that they are available if the task is revived.
The full tables are in `docs/pilot_report.md`, and the raw outputs are in `results/pilot/`. The numbers are from the sandbox, and nothing
here has been run on the laptop.

## The generator and the model

All generator checks passed. The key-blind and no-context baselines match theory. Shuffling the context leaves a model's output unchanged,
and a model without access to the context scores at chance.

## Learning rate dominates the dot-product result at small head size

At P = 64 and D = 3, the tied dot-product arm scored 0.513, 0.678, 0.690, 0.997, 0.999 and 1.000 at learning rates 1e-3, 3e-3, 1e-2, 3e-2,
1e-1 and 3e-1. At P = 32 and D = 2 it scored 0.327, 0.378, 0.555, 0.930, 0.963 and 0.979. A run of 12000 steps at 1e-3 reached only 0.692.
The L^p arms stayed between 0.98 and 1.0 at every rate. The untied dot-product arm collapsed at 0.1 and 0.3. The gap seen at a low shared
rate was therefore mostly a gap in optimisation speed. A single shared rate is either unfair to the dot arms or close to the stability
edge of the untied arm.

## The transition at learning rate 3e-2

For D of 3 and above, every arm is saturated up to P = 64. At D = 2 the tied dot product scored 0.93, 0.95, 0.93 and 0.43 at P = 8, 16,
32 and 64. The untied arm scored 0.94, 0.94, 0.91 and 0.86. L1, L2 and L-infinity scored 1.000 throughout. D = 1 is degenerate, because
all L^p arms are the same function there. With 1024 keys: at D = 3 the tied dot product reached 0.985 at P = 128 and 0.336 at P = 256,
while L2 reached 1.000 at both. At D = 2 the tied dot product reached 0.330 and 0.279 while L2 reached 1.000 and 0.986. Run time for L2
grew from 65 s at P = 32 to 2600 s at P = 256.

## The primary outcome could not be computed as designed

The proposed outcome was the pair count P at which accuracy falls below 90% at a fixed D. The L^p arms are at the ceiling, so that P exists
only for the dot arms and only within P <= 64. The values of p cannot be told apart by final accuracy. Steps to reach 99% accuracy were 320
to 500 for the L^p arms and were never reached by the tied dot product, but at 100-step resolution these show no clear order among the p.

## Why the task is easy for distance scores

A query and its matching context token share the same key embedding and differ only in the value part. A distance score can find a match at
distance zero for every p. This is a property of the task, not of the metric.

## Seed variance

Five paired seeds at lr 3e-2: at (P = 64, D = 2) the tied dot product scored 0.850 (sd 0.0069) and the paired difference of the L^p arms
over it was +0.150 (sd 0.0069). At (P = 32, D = 2) it scored 0.926 (sd 0.012) and the difference was +0.074 (sd 0.012).

## Epsilon

With r = 0 the gradient is NaN for p > 1. Accuracy degrades for p = 8 at r >= 1e-6. The proposed r is 1e-12. The training-sensitivity runs
are in `epsilon_runs`. The direct p = 2 kernel was used because the matmul expansion lost accuracy at D = 1.

## Lessons for a revival

1. Choose the learning rate for each arm and each D before comparing arms, or report accuracy across rates. A shared low rate makes the
   distance arms look better than they are.
2. A task where a match sits at distance zero cannot rank the values of p. A task with a harder matching rule would be needed for that.
3. The cost per run grows quickly with P (190 s at P = 64 against 20 s at P = 8 for the pilot settings).
