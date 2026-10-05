# Metric self-attention: summary of the original proposal

**Status: RECORD.** Recorded 2026-10-05. This is the broader proposal as it stood after the pilot, before the project was refocused on head size and the choice of p (see `docs/project_specification.md`). It is kept as a record of what the planned project was. The authoritative design decisions are the lines tagged [stated] in the project memory guide.

This summary is for someone who has not seen the project. It separates what is decided from what the pilot found and from what is in question. Every term is defined where it first appears.

## 1. The research question

A transformer is a neural network that predicts the next item in a sequence. Its core operation is **self-attention**. For each position in the sequence, the model decides how much to look at every earlier position, and then blends information from those positions according to those weights.

Each position produces three vectors. The **query** says what this position is looking for. The **key** says what each earlier position offers. The **value** is the content that gets blended in. The model compares the query with every key and produces a **score** for each pair. A **softmax** turns the scores into weights that are positive and sum to one. The output is the weighted sum of the values.

In almost every transformer, the score is the **dot product** of query and key, a sum of coordinate-wise products. This project asks what happens if the score is instead the negative **distance** between query and key. Closer pairs then score higher.

The question is whether this change affects what a transformer can learn.

## 2. The proposed change

**L^p distance.** The distance between two vectors uses a parameter p. Subtract the vectors, raise the absolute value of each coordinate difference to the power p, sum, and take the p-th root. For p = 2 this is ordinary straight-line distance. For p = 1 it is the sum of absolute differences, also called taxicab distance. For p = infinity it is the largest single coordinate difference, also called Chebyshev distance.

**The grid.** The project tests p in {1, 1.5, 2, 3, 4, 8, infinity}.

**Tied projections.** The query and key come from the same learned matrix, written W_Q = W_K. This makes the distance a true metric on the vectors. It also means a position is at distance zero from itself, so the diagonal is masked. Each position may not attend to itself, except position 0, which has nothing else to attend to.

**The score formula.** For the distance arms, the score is -beta * c * distance. For the dot arm, it is beta * c * (query . key). In both formulas, beta is a learned scale per head, and c is a fixed constant calibrated once at the start. It is chosen so that the spread of scores at initialisation is the same for every arm. This keeps the comparison to the score's shape, not its size. The write-up must say that the scale was not tuned per arm.

**Epsilon.** A tiny number epsilon is added inside the root so the gradient is finite when two vectors coincide. It is set relative to the typical size of the sum inside the root. The proposed ratio is 1e-12.

**The nine arms.** The comparison has nine versions of the model, called arms:
- tied dot product, the control for everything else;
- seven distance arms, one for each p;
- untied dot product, with separate query and key matrices as in a standard transformer. It sits outside the main design and measures the cost of tying.

**Rules from the design stage.** No normalisation of queries and keys. Learned absolute position embeddings, not rotary ones. The L^p metric depends on the coordinate axes when p is not 2, and the project accepts that.

## 3. The two tasks

**Task 1: recall (MQAR).** MQAR stands for multi-query associative recall, a synthetic task from the literature. Each sequence has P key-value pairs, called the context. After the context come Q = P/2 queries, each of which shows a key from the context and asks for its value. There are 256 possible keys and 64 possible values. Chance accuracy is 1/64.

The model has one attention layer, one head, no feed-forward block, and a width (d_model) of 64. A query attends only to the context pairs. The head dimension D is the size of the space where queries and keys are compared, and it is swept. The project measures accuracy as a function of P and D.

**Task 2: character-level language model.** The corpus is Tiny Shakespeare, about 1.1 million characters with 65 distinct symbols, split 90/10 into training and validation. The model has 2 layers, d_model 64, 4 heads of dimension 16, a feed-forward block in each layer, a context of 64 characters, and about 100 thousand parameters. The metric is **validation bits per character (bpc)**, the average information the model needs per character on held-out text. Lower is better.

## 4. Protocol and statistics

- All arms use the same data and the same starting values for shared parameters, matched by **seed** (the number that fixes randomness). Comparisons are paired: for each seed, one arm's score minus another's.
- About 5 seeds per setting. Seven distance arms are each compared with the tied dot control, with a Holm correction for multiple testing.
- The primary outcomes are the pair count P at which recall accuracy falls below 90% at a fixed D, and validation bpc for the language model.
- Training uses the AdamW optimiser, 3000 steps for recall and 2500 for the language model. Final weights are saved from every run.
- The budget is 48 hours on a 2020 Intel MacBook. The custom distance routine is written to save memory.

## 5. State of the work (as of the pilot)

The code, tests and a single entry-point script are written, and the pilot has been run. The pilot report is in `docs/pilot_report.md`. The full runs have not started.

## 6. What the pilot found

- **Recall at a low learning rate favours distance, but that is mostly optimisation.** At a learning rate of 1e-3 the dot arm scored 0.51 at (P = 64, D = 3), against 0.99 for the distance arms. At 3e-2 the dot arm reached 0.997. The gap was largely a matter of training speed.
- **At an adequate rate the distance arms are at the ceiling.** For D of 3 and above and P up to 64, all arms score nearly 1.0. At D = 2 the dot arm drops (0.85 at P = 64) while all seven distance arms stay at 1.000. The distance arms cannot be told apart from each other by final accuracy.
- **The match is easy by construction.** In this task a query and its matching context token share the same key embedding, so a distance score finds them at distance zero.
- **Convergence speed also shows no clear ordering among p** at 100-step resolution. There is a weak, single-seed hint at low rates that mid values of p learn faster and L1 and L-infinity slower.
- **The language model shows no separation.** With 5 seeds at a learning rate of 3e-3, the paired difference between L2 and dot is -0.005 bpc (noise sd about 0.016). L-infinity is 0.017 worse. The smallest detectable difference is about 0.02 to 0.03 bpc. Only 3 of the 9 arms were trained to full length.
- **The models underfit.** Training loss is about 2.13 bpc against 2.41 on validation. A good model of this kind would reach about 1.5 to 1.7.

## 7. What was in question

1. **Recall as a test.** The distance arms saturate, so the primary recall outcome cannot be computed for all arms at one D. The learning-rate confound also needs a rule.
2. **The language model as a test.** At head dimension 16 the arms are indistinguishable. Possible reasons were listed (roomy heads, attention not the bottleneck at this scale, a recall-specific difference diluted in the average, a moderate logit range). None was tested.
3. **A broader task.** A "planted-metric" classification task was proposed and judged too contrived. It was not adopted.
4. **A small-head language model.** Shrinking only the query-key dimension to 2 or 4, plus a uniform-attention control, was discussed. This became the basis of the refocused proposal.

Decisions that were awaiting the user: the recall learning rate and primary outcome, the value of epsilon, the direct kernel for p = 2, the target score spread, the language-model learning rate and seed count, laptop or cloud, context length, and whether to commit the saved weights.
