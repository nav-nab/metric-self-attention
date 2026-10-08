# Metric self-attention

Does replacing the query-key dot product in self-attention by a negative L^p distance score change what a transformer can learn?
The project studies how the head size D (the query-key dimension per head) interacts with the choice of p. The main experiment is a
small character-level language model on Tiny Shakespeare, run over head sizes D = 2, 4, 8, 12 and 16 with nine arms: the tied dot
product, seven L^p distances (p = 1, 1.5, 2, 3, 4, 8, infinity) and the untied dot product. A uniform-attention control is a full arm too, and an unmasked untied dot product is run at D = 16 as a standard-transformer reference. The context is 64 characters, and each run takes 4,000 steps. Each arm is given its own learning rate at each head size, chosen by a one-seed scan. Five seeds per cell, run on the laptop, with the final weights committed. The primary analyses are the D x p interaction and the trend of the gap to the tied dot product across D.

**Status (2026-10-08): the experiment has been run and the results are ready for analysis.** The design was settled on 2026-10-05, and the code was written and tested. The full run (141 + 46 scan runs and 235 main runs) finished on 2026-10-07 without a divergence. `docs/results_guide.md` describes the results. All design issues are
settled, and one detail is deferred. Section 12 of `docs/design_details.md` lists the decisions.
The first version of the code and its pilot are kept in `shelved/` as a record and are not part of the live project.

* `docs/project_outline.md`: the overview of the project (question, background, motivation, experiment). Read this first.
* `docs/design_details.md`: the settled decisions and the reasoning behind them. Read this second.
* `docs/implementation_plan.md`: how the code is organised, including interruption and resumption, with the log of coding decisions in section 13 and the record of the run in section 14. Read this third.
* `docs/results_guide.md`: the record of the run, the files that it produced, the points to keep in mind when interpreting them, and the working rules for the agent that analyses the results. Read this fourth.
* `execute_project.py` and `msa/`: the code. `python execute_project.py <stage>` runs a stage (check, smoke, calibrate, scan, select, main, analyze, all). `tests/` holds the unit tests.
* `results/`: calibration constants, scan and main runs, `selected_rates.json`, `scan_table.csv` and `analysis/` (tables and figures).
* `data/tinyshakespeare.txt`: the corpus (1,115,394 characters, md5 `6fb458f1232090904fb40fe944165e91`).
* `requirements.txt`: the pinned environment (torch 2.2.2, numpy below 2).
* `shelved/`: the first version. See `shelved/README.md`.

The design decisions are kept in the project guide with the research notes. Nothing is run without the user's agreement.
