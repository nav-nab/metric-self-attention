# Metric self-attention: head size and the choice of p

**Status: DRAFT, design phase.** Last updated 2026-10-05. This is the proposal centred on the interaction between head size and the choice of p. The project is being rebuilt from scratch against it, and no code or run exists yet. The issues are being settled one at a time, and section 7 lists what is settled and what is open. Nothing in this document is a final plan until section 7 says so. The earlier version of the project is kept in `shelved/` as a record.

This document is for a reader who has not seen the project. It keeps the pilot results that bear on the central question and leaves out the rest.

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
- **Learning rate.** The best learning rate may depend on D and on the arm, and this is a real risk. In the earlier work, the dot arm needed a far higher rate at small D, and a shared low rate gave the false impression that distance was better. Each arm therefore gets its own rate at each D, chosen by a scan on a fixed grid. The protocol is described in the Notes.

**Training length.** Each run takes 4,000 optimiser steps.

**The outcome.** Validation bpc for each (D, arm) cell. The analysis looks at the whole surface: how the gap to the dot product changes with D, and how the ranking of p changes with D.

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

## 7. Design status

This section is the checklist for the design phase. The Notes below give the reasoning for some of the settled points.

**Settled** (the user's decisions; dates are 2026-10-05 unless given).
- The question: how head size D and the choice of p interact, studied on the language model alone.
- The arms: the tied dot product as the control, the seven L^p arms (p = 1, 1.5, 2, 3, 4, 8, infinity), and the untied dot product as a reference. This nine-arm grid is confirmed. The uniform control is added to it, and its status is open (below).
- The head-size grid: D = 2, 4, 8, 12 and 16. Only the query-key dimension changes, and the value width stays at 16 per head. D = 1 is skipped.
- Queries and keys are tied in every arm except the untied reference. The diagonal is masked for all arms, and position 0 may attend to itself.
- The score is minus the distance, scaled by a learned factor per head (initial value 1) and a frozen calibrated constant. The constant is chosen so that the standard deviation of the scores equals 1 at initialisation, and it is recomputed for every head size, arm and layer. Matching at initialisation is the whole scheme, and the scale is not tuned per arm.
- The epsilon inside the root is relative to the typical size of the sum inside the root. The ratio is r = 1e-12 (decided 2026-10-05). The p = infinity distance is the largest coordinate difference, with no smoothing.
- No normalisation of queries and keys. Learned absolute positions. The dependence of non-L2 metrics on the coordinate axes is accepted and flagged.
- The model: two layers, width 64, four heads, MLP width four times the width, context 64 characters for every arm, tied input and output embeddings. The corpus is Tiny Shakespeare split 90/10, and the outcome is validation bpc.
- One extra reference arm, the untied dot product without the diagonal mask, is run at D = 16 with the same five seeds (decided 2026-10-05).
- The p = 2 distance is computed from direct coordinate differences, as for every other p, and not from the matrix-multiply expansion (decided 2026-10-05).
- Each run takes 4,000 steps (decided 2026-10-05).
- The learning-rate protocol: a one-seed scan at the final length, then each arm at each D is run at its own best rate from the scan. The details are in the Notes (decided 2026-10-05).
- The final weights of every run are saved and committed to git (decided 2026-10-05).
- Five seeds per cell. The uniform control is a full arm with the same five seeds. The runs are made on the laptop (decided 2026-10-05).
- The recall task (MQAR) is shelved, and the language model is run by itself first. The project is restarted from scratch.

**Open.** Each point will be settled with the user, in an order the user chooses.
- The statistical plan. The earlier plan paired arms by seed and compared each L^p arm with the tied dot product under a multiple-comparison correction. It has not been re-confirmed for the new design.

## 8. Possible future directions

These are not part of the planned experiment. They are recorded so that they can be picked up later, and none has been designed or costed in detail.

- **Relative position bias.** Replace the learned absolute positions by a learned bias on the logit that depends on the distance between two positions. Run a reduced set of arms (the tied dot product and p = 1, 2 and infinity) with a few seeds. This would test whether the conclusions depend on the choice of absolute positions, which is a stated limitation. The bias adds its own spread to the logits, so the calibration would need rethinking. The user dropped this run from the project on 2026-10-05.
- **A longer context.** Repeat a reduced set of arms at a context of 128 characters, for example the tied dot product, L2 and L-infinity at D = 2 and 4. A longer context gives each query more keys, which could widen any gap at small D.
- **Normalised queries and keys.** An ablation with query-key normalisation, to be run only if some arm turns out to be unstable.
- **The recall task.** The shelved recall task (MQAR) could be revived after the language-model results are seen, if a controlled setting for the head-size effect is wanted.
- **Savings at inference.** If a distance head works at a very small D, the key cache could be smaller at inference time.

## Notes

**Why the diagonal is masked (decided; confirmed 2026-10-05).** With tied projections, a position's query and its own key are the same vector. The distance between them is exactly zero, which is the best score a distance can give, so an unmasked position would favour itself in every layer. For the tied dot product the self-score is the squared vector length, and the Reformer paper (Kitaev, Kaiser and Levskaya, ICLR 2020, section 2) notes that this "will almost always" exceed the score with any other position. That paper is the source of the masking rule and of its exception: a token may not attend to itself unless it has no other valid target, such as the first token. We follow it, so position 0 may attend to itself and every other position may not. The Reformer argument covers only the dot product, and the argument for the distance is the project's own geometric reasoning. A second reason is calibration: the self-key would inflate the measured spread of scores and distort the starting scale. The cost of masking is small, because the residual connection already carries a position's own content to the next layer. The rule applies to all arms in the language model, including the untied dot product, so that the comparison isolates the score function. It does not apply to the recall task, where each query attends only to the context pairs. The write-up must state that this masking departs from a standard language model, so the dot-product baseline is slightly non-standard. An extra reference arm, the untied dot product without the mask, is run at D = 16 to measure the size of this departure.

**Calibration of the score scale (decided; closed 2026-10-05).** Different score functions give different spreads of scores across the keys. The softmax is sensitive to that spread, so an uncorrected comparison would partly measure how sharp each arm's attention happens to be at the start. Each arm therefore multiplies its score by a fixed constant, chosen so that the standard deviation of the scores across a query's allowed keys equals a common target of 1 at initialisation. The constant is measured once for each combination of head size, arm and layer, so it is recomputed for every head size in the experiment. The measurement uses a model built with a fixed calibration seed on fixed batches of text, and all seeds share the result. It excludes the masked diagonal and the early positions with fewer than about 8 allowed keys, and it proceeds layer by layer. A learned scale per head, starting at 1 for every arm, sits on top of the constant, so the model can sharpen or flatten its attention during training. Matching at initialisation is the whole scheme. We do not match the arms later in training, and we do not tune the scale per arm. The target of 1 is a convention that gives soft but not flat attention at the start, and no other target has been tested. A single standard deviation is a crude summary of the score distribution, because distance scores are more skewed than dot-product scores, and this is accepted. The calibrated constant for the tied dot product should land near one over the square root of the head dimension, which serves as a sanity check. The write-up must state that the scale was not tuned per arm and that the target was chosen by convention.

**Shelving of the recall task (decided 2026-10-05).** The user shelved the recall task (MQAR) because the distance arms sit at its accuracy ceiling and the dot-product result depends on the learning rate, and because further pilots on it would not end. The code, the design decisions and the pilot findings were moved into `shelved/pilot_v1/mqar/`, with its own entry point and tests, and the main engine was made task-agnostic so that the task plugs in. The language model is run by itself first. The full results then decide whether the recall task is needed.

**Restart from scratch (decided 2026-10-05).** The user shelved all of the version 1 code, including the language-model code, the pilot report and the pilot results, in `shelved/pilot_v1/`, and the project is being rebuilt from scratch against this specification. The version 1 design record is `shelved/pilot_v1/DESIGN.md`. It marks which decisions still hold and which were unapproved implementation choices (r = 1e-12 and the direct p = 2 kernel). The grid of nine arms was confirmed on the same day.

**Head-size grid (decided 2026-10-05).** The grid of head sizes is D = 2, 4, 8, 12 and 16. The user added D = 8 and D = 12 to the earlier first round of D = 2 and 4 beside the default D = 16.

**Learning-rate protocol and training length (decided 2026-10-05).** The earlier work found that the best learning rate can depend on the head size and on the arm, and that a shared rate can manufacture a difference between arms. The protocol therefore tunes the rate separately for every arm and head size, in two stages. The first stage is a scan. Every arm is run at every head size, once each, at the rates 3e-3, 1e-2 and 3e-2, for the full 4,000 steps. The scan seed is not one of the five final seeds. The two extra arms, the uniform control and the unmasked untied dot product, are scanned in the same way. Each arm and head size then takes the rate with the lowest validation bpc. If the best rate is at either end of the grid, that arm and head size is run at one more rate, a factor of 3 beyond the grid. The second stage is the main experiment. Each arm and head size is run for the five final seeds at its chosen rate. The scan is part of the experiment, its results are reported, and the protocol is fixed in advance. The chosen rates and the full scan table go in the write-up. A run that diverges at a rate is reported as a finding, and that rate is not chosen. One seed has noise of about 0.01 bpc. Where two neighbouring rates differ by less than that, the choice is arbitrary but harmless, because the loss is flat there. The write-up must say two things. The first is that each arm's rate was tuned, so the arms differ in more than the score function. The second is that the rate was chosen on the validation set that the results are reported on. The cost is about 8 hours for the scan and about 14 hours for the main runs at 4,000 steps on the laptop, which is about 23 to 24 hours in total with the extra arms. These are estimates with an uncertainty of 30 to 40 percent.

**Epsilon and the p = 2 kernel (decided 2026-10-05).** A small number eps is added to the sum inside the root, so that the gradient stays finite when two vectors coincide. Without it, the gradient is not finite for every p above 1 at an exact match. The number is relative: eps is r times the typical size of the sum for that arm, layer and head size, measured at initialisation and then frozen. The ratio is r = 1e-12. The earlier work found that r = 0 fails, that r = 1e-12 is safe in float32, and that larger values distort the metric for large p. The p = 2 distance uses direct coordinate differences, like every other p. The matrix-multiply expansion is faster, but it lost accuracy when distances are tiny, and the project does not use it. This makes the L2 arm about three times more expensive than it would be with the expansion. The cost estimates already include this. The value r = 1e-12 distorts the metric slightly for large p, because it puts a floor under the distance. By the formula, the floor is the fraction r^(1/p) of the typical distance scale. That is about 3 percent at p = 8, 0.1 percent at p = 4 and 0.0001 percent at p = 2. It is smaller for the other finite p. The p = infinity arm has no epsilon and no floor. The write-up must state the floor for p = 8.
