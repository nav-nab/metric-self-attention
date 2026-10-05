# Metric self-attention: head size and the choice of p

**Status: design settled, implementation starting.** Last updated 2026-10-05. All design issues are settled, and one detail is deferred (an equivalence margin that is needed only if an equivalence claim is made). The project is being rebuilt from scratch against this design. No code or run exists yet, and the code is written against `implementation_plan.md`. The earlier version of the project is kept in `shelved/` as a record.

This is the first of two documents. It gives the question, the background, the motivation and the shape of the experiment. The second, `design_details.md`, gives the settled decisions and the reasoning behind them, and should be read after this one. This outline is written for a reader who has not seen the project. It keeps the pilot results that bear on the central question and leaves out the rest.

## 1. The question

Attention is the part of a transformer that decides which earlier positions in a sequence a given position should look at. The usual way to make that decision is a dot product between two vectors. This project replaces the dot product with a negative distance and asks when that helps.

The distance has a free parameter p, and the attention has a size parameter, the **head dimension** D. Each head compares queries and keys in a small space of D dimensions. The project studies how these two factors interact. For each head size D, which choice of p works best, and does any choice of p beat the dot product? Does the answer change as D shrinks?

## 2. Background and terms

**Attention.** Every position produces a **query**, a **key**, and a **value**. The model scores each query against each earlier key. A **softmax** turns the scores into positive weights that sum to one. The output is the weighted blend of the values.

**Head.** A transformer layer runs several independent attention computations in parallel, and each one is a head. The head dimension D is the length of each query and key vector. A small D means that queries and keys live in a small space.

**The dot-product score.** The standard score is the sum of coordinate-by-coordinate products. Geometrically, a query prefers the keys that lie furthest in its direction. The preferred region is a half-space, which means everything on one side of a flat boundary.

**The distance score.** The alternative score is minus the distance between query and key. Closer keys score higher, so the preferred region is a ball around the query.

**L^p distance.** To compute it, subtract the two vectors, raise each coordinate difference to the power p in absolute value, add them up, and take the p-th root. For p = 2 this is straight-line distance. For p = 1 it is taxicab distance, the sum of absolute differences. For p = infinity it is the largest single coordinate difference. The ball is a diamond for p = 1, round for p = 2, and a box for p = infinity. The project uses p in {1, 1.5, 2, 3, 4, 8, infinity}.

**Tied projections.** The query and key are produced by the same learned matrix. This makes the distance a genuine metric on the vectors, and it is a design decision of the project. As a consequence every position is at distance zero from itself, so the diagonal is masked.

**Calibration.** The score is scaled by a learned factor per head and a fixed constant. The constant is set at the start so that all arms begin with the same spread of scores. This isolates the shape of the score from its size. The scale is not tuned per arm, and the write-up must say so.

## 3. Why head size and p might interact

Three geometric facts motivate the design. They are reasoning, not results.

**A small head limits the dot product more than it limits the distance.** Suppose a head must pick out one specific key from many. The dot product can single out a key only if that key sits at the edge of the cloud of keys, because the preferred region is a half-space. With few dimensions there is little room at the edge. For keys arranged evenly on a circle, the score margin between the best key and its nearest rival shrinks much faster with the number of keys for the dot product than for the distance. A distance score can pick out a key anywhere, including the middle, because a ball around the query can sit anywhere. So the distance is expected to hold up better as D shrinks.

**Some choices of p become the same metric in low dimension.**
- In one dimension all values of p are the same function.
- In two dimensions, p = 1 and p = infinity are the same metric up to a 45-degree rotation and a scale. The learned projection can supply that rotation. So at D = 2 these two arms have equal expressive power, and any difference between them reflects optimisation, such as the sparser gradients of the p = infinity metric.
- In higher dimensions the metrics differ, but a large head gives the projection enough freedom to reshape the geometry. A large head may therefore absorb differences in p as well.

**The expected pattern.** If both facts hold, the effect of p would be largest at intermediate D. It would vanish at D = 1, be partial at D = 2, and be absorbed at large D. The advantage of distance over dot would be largest at small D and would fade as D grows. This is a hypothesis for the experiment to test.

## 4. Proposed experiment

**The factors.** Head dimension D and the arm. The arms are:
- tied dot product, as the control;
- the seven L^p arms;
- untied dot product, with separate query and key matrices, as a reference that shows the cost of tying;
- an **unmasked untied dot product**, run at D = 16 only. This is the standard transformer, with separate query and key matrices and no diagonal mask. It measures how far the masked baseline departs from a standard model;
- a **uniform control**, in which every allowed key gets the same weight. It is not a real candidate. It shows how much of the result attention can move at all, and so gives scale to every difference. It is a full arm and gets the same number of seeds as the others.

**The main task.** A small character-level language model on Tiny Shakespeare, about 1.1 million characters from Shakespeare's plays. The model is a two-layer transformer of width 64, with four heads and a context of 64 characters, the same for every arm. The score is **validation bits per character (bpc)**, the average information the model needs per character on held-out text. Lower is better.

**Varying D.** Only the query-key dimension of each head is shrunk. The value width stays at 16 per head, so the rest of the model is unchanged. The grid is D = 2, 4, 8, 12 and 16, where D = 16 is the default head size of the model (width 64 divided by 4 heads). D = 1 is skipped, because all L^p arms coincide there.

**The recall task is shelved.** A synthetic recall task (MQAR) was considered as a controlled setting where head size and context size interact. It is shelved as of 2026-10-05. Its code, design decisions and pilot findings are in `shelved/pilot_v1/mqar/`. The language model is run by itself first, and the full results decide whether the recall task is needed.

**Fairness safeguards.**
- **Pairing.** Every arm with the same seed (the number that fixes randomness) sees the same data and starts from the same shared parameters, so differences are paired.
- **Replication.** Five seeds per cell.
- **Learning rate.** The best learning rate may depend on D and on the arm, and this is a real risk. In the earlier work, the dot arm needed a far higher rate at small D, and a shared low rate gave the false impression that distance was better. Each arm therefore gets its own rate at each D, chosen by a scan on a fixed grid. The protocol is described in `design_details.md`, section 8.

**Training length.** Each run takes 4,000 optimiser steps.

**The outcome.** Validation bpc for each (D, arm) cell. The analysis looks at the whole surface: how the gap to the dot product changes with D, and how the ranking of p changes with D. The statistical analysis treats the seeds as blocks. Its two main questions are whether the effect of p depends on D, and how the gap between each L^p arm and the tied dot product changes with D. Secondary analyses report the gap in each cell, the shape of the effect of p, and how often each p is best. The full plan is in `design_details.md`, section 9.

## 5. Relevant results from the pilot

These come from version 1 of the project, which is shelved in `shelved/pilot_v1/`. Each finding is listed there, with its caveats and its relevance to this project, in `shelved/pilot_v1/PILOT_FINDINGS.md`. The full report is `shelved/pilot_v1/docs/pilot_report.md`. Every pilot run used head size 16, and no pilot number describes a small head.

- At head dimension 16, the language-model arms are within seed noise of each other. Tied dot and L2 are indistinguishable, and p = infinity is slightly worse. This is why the project now turns to small heads.
- Background (recall task, now shelved): on the recall task, small head dimension is where the dot product breaks down while the distance arms do not. At larger D every arm solves the task. The distance arms are all near perfect, so the recall task cannot rank the values of p.
- At an adequate learning rate, convergence speed among the L^p arms shows no clear ordering. There is a weak hint that mid values of p learn faster early and that p = 1 and p = infinity learn slower. It rests on one seed.
- Learning rate matters. The dot arm is more sensitive to it at small D than the distance arms.
- The seed-to-seed spread of validation bpc is about 0.01, and the smallest difference detectable with five seeds is about 0.02 (finding 7 in the pilot findings).
- The language model underfits: training loss is well above what a model of this kind could reach. So these experiments probe a regime limited by capacity as much as by attention.

## 6. Limitations

- Heads of dimension 2 or 4 are far smaller than anyone uses in practice. The point is to find where the metrics differ. If a distance head works at tiny D, one practical angle is a smaller key cache at inference time, to be raised only if the data support it.
- The model is small and underfit, so an effect of the score may be hidden by other limits.
- Non-L2 metrics depend on the coordinate axes. This is accepted and flagged.
- Seed noise is about 0.01 bpc, so only differences above roughly 0.02 are detectable with five seeds. This figure comes from an earlier version at 2,500 steps and D = 16, and it will be re-measured.
- Each arm's learning rate is chosen on the same validation set that the results are reported on. The selection uses a different seed from the final seeds, and every arm gets the same treatment, so the bias is expected to be small. It is not zero.
- The tie and the diagonal mask apply to most arms and may shape results more than the score does.

## 7. Possible future directions

These are not part of the planned experiment. They are recorded so that they can be picked up later, and none has been designed or costed in detail.

- **Relative position bias.** Replace the learned absolute positions by a learned bias on the logit that depends on the distance between two positions. Run a reduced set of arms (the tied dot product and p = 1, 2 and infinity) with a few seeds. This would test whether the conclusions depend on the choice of absolute positions, which is a stated limitation. The bias adds its own spread to the logits, so the calibration would need rethinking. The user dropped this run from the project on 2026-10-05.
- **A longer context.** Repeat a reduced set of arms at a context of 128 characters, for example the tied dot product, L2 and L-infinity at D = 2 and 4. A longer context gives each query more keys, which could widen any gap at small D.
- **Normalised queries and keys.** An ablation with query-key normalisation, to be run only if some arm turns out to be unstable.
- **The recall task.** The shelved recall task (MQAR) could be revived after the language-model results are seen, if a controlled setting for the head-size effect is wanted.
- **Savings at inference.** If a distance head works at a very small D, the key cache could be smaller at inference time.

The settled decisions, the open items and the reasoning behind each decision are in `design_details.md`.
