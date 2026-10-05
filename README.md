# Metric self-attention

Does replacing the query-key dot product in self-attention by a negative L^p distance score change what a transformer can learn?
The project studies how the head size D (the query-key dimension per head) interacts with the choice of p. The main experiment is a
small character-level language model on Tiny Shakespeare, run over head sizes D = 2, 4, 8, 12 and 16 with nine arms: the tied dot
product, seven L^p distances (p = 1, 1.5, 2, 3, 4, 8, infinity) and the untied dot product. A uniform-attention control is a full arm too, and an unmasked untied dot product is run at D = 16 as a standard-transformer reference. The context is 64 characters, and each run takes 4,000 steps. Each arm is given its own learning rate at each head size, chosen by a one-seed scan. Five seeds per cell, run on the laptop, with the final weights committed. The primary analyses are the D x p interaction and the trend of the gap to the tied dot product across D.

**Status (2026-10-05): design settled, implementation starting.** The project is being rebuilt from scratch, and no code or run exists yet. All design issues are
settled, and one detail is deferred. Section 12 of `docs/design_details.md` lists the decisions.
The first version of the code and its pilot are kept in `shelved/` as a record and are not part of the live project.

* `docs/project_outline.md`: the overview of the project (question, background, motivation, experiment). Read this first.
* `docs/design_details.md`: the settled decisions and the reasoning behind them. Read this second.
* `docs/implementation_plan.md`: how the code will be organised, including interruption and resumption. The brief for the code, with working rules for the implementing agent. Read this third.
* `data/tinyshakespeare.txt`: the corpus (1,115,394 characters, md5 `6fb458f1232090904fb40fe944165e91`).
* `requirements.txt`: the pinned environment (torch 2.2.2, numpy below 2).
* `shelved/`: the first version. See `shelved/README.md`.

The design decisions are kept in the project guide with the research notes. Nothing is run without the user's agreement.
