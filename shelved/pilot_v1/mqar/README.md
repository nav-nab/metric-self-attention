# Shelved: the MQAR recall task

**Status: SHELVED on 2026-10-05, by the user's decision.** Nothing in this folder runs unless you call it. The main project now runs the
language model by itself. The user will look at the full language-model results first and then decide whether this task is needed.

## Why it was shelved

The recall task was meant to be a controlled setting where head size and the choice of p visibly interact. The pilot showed two problems.
The first problem is that the distance arms sit at the accuracy ceiling for every p, so the task cannot rank the values of p. The second
problem is that the dot-product arms need a much higher learning rate at small head sizes, so the comparison depends on a rate choice that
the task itself does not settle. The user judged that continuing would lead to a chain of further pilots, and chose to wait for the
language-model results. The numbers are in `PILOT_FINDINGS.md`.

## What is in this folder

| File | Content |
|---|---|
| `DESIGN.md` | Every design decision for the task, and which shared decisions of the main project it relies on. |
| `PILOT_FINDINGS.md` | What the MQAR pilot found and what it taught. |
| `gen.py` | The data generator (a vectorised version and a readable reference version), the fixed evaluation sets, the baselines and the generator checks. |
| `model.py` | The recall model: one attention layer, one head of dimension D, no MLP, no positions. |
| `evaluate.py` | Accuracy and diagnostics, and the two sanity checks (shuffled context, no context). |
| `task.py` | Registers the task `mqar` with the training engine (`msa.train`). Importing it is what makes `mqar` available. |
| `pilot.py` | The recall stages of the original pilot, re-pointed at the plug-in task. |
| `execute_mqar.py` | The entry point for those stages. |
| `tests/` | The tests for the generator and the calibration constant. |

The original pilot outputs are still in `results/pilot/` (the MQAR ones are the `transition*`, `lr_arms`, `epsilon_runs`, `lr_safeguard`,
`seedvar` and `lr_scan` outputs there). New runs from this folder go to `results/mqar/` by default.

## How the task plugs into the engine

The engine in `msa/train.py` knows nothing about recall. A task is a subclass of `msa.train.Task` that is registered by name. A config
carries the field `task_module = "mqar.task"`, so a worker process started with `spawn` can import the task before it builds the
run. The score module, the arms, the calibration routine and the shared-initialisation rule all come from `msa`.

## How to revive it

1. Run the shelved tests: `python mqar/execute_mqar.py tests`.
2. Run a quick check of one stage: `python mqar/execute_mqar.py generator`.
3. Decide the open points listed at the end of `DESIGN.md` before launching anything larger. The main ones are the learning rate and the
   primary outcome.
4. Re-check the shared decisions in `DESIGN.md` against the current specification (`docs/project_specification.md`), since the main
   project may have moved on.

Equivalence note: when the engine was split into a task-agnostic core and plug-ins, the language-model runs and the recall runs were both
compared against the old combined engine. Weights, results and run ids were bit-identical.
