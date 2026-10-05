# Shelved: version 1 of the project (code, pilot and results)

**Status: SHELVED on 2026-10-05, by the user's decision.** The live project is restarted from scratch. This folder is a complete,
self-contained snapshot of the first version. It is kept so that its design record, its findings and its tested pieces can be
consulted. Rather than adapting it, the new version is written fresh.

## Why it was shelved

The first version was designed around two tasks (the recall task MQAR and the language model) at a single head size, and its pilot
answered the questions of that design. The project has since moved on in several ways. It is now centred on the interaction between
head size D and the choice of p. It runs the language model by itself, over a grid of head sizes (D = 2, 4, 8, 12, 16) with a
uniform-attention control. Calibration is redone for every head size. Several decisions were also settled or reopened (the masked
diagonal, the target spread of 1, the learning-rate protocol per head size). The user judged that adapting the old code to all of
this would be harder than starting again, and that the pilot's numbers describe an earlier version of the design.

## What is in this folder

| Path | Content |
|---|---|
| `DESIGN.md` | The design record of version 1: what was decided, by whom, and what has since changed. |
| `PILOT_FINDINGS.md` | The pilot findings that concern the language model and the shared machinery, and which of them still matter. |
| `docs/pilot_report.md` | The full pilot report as it was delivered on 2026-10-02, with its glossary. |
| `docs/implementation_notes.md` | Exactly what the code does (score, calibration, models, training protocol, task plug-ins). |
| `msa/` | The code: distance kernel, gradient checks, models, calibration, task-agnostic training engine, benchmarks, pilot stages. |
| `mqar/` | The shelved recall task, with its own README, design record, findings, entry script and tests. |
| `tests/` | The tests of the main code (9 tests). |
| `execute_project.py` | The single entry point for the language-model stages. |
| `results/pilot/` | The raw pilot outputs: one JSON file per stage and one folder per run, with final weights (`final.pt`) kept on disk. |
| `results/pilot_vm/` | Timings measured in the Linux VM of the desktop app. |

## How to run it

Run everything from inside this folder, with the pinned environment (`../../requirements.txt`).

```
cd shelved/pilot_v1
python execute_project.py tests          # 13 tests: the main ones and the recall ones
python execute_project.py env            # environment check and smoke training steps
python execute_project.py -h             # list of stages
python mqar/execute_mqar.py -h           # the recall stages
```

The corpus is found in the `data/` folder of the repository, two levels up. Output goes to `results/` in this folder. The
`final.pt` weight files and checkpoints are not tracked by git.

## State at the time of shelving

* The pilot stages `env`, `gradcheck`, `bench`, `calibrate`, `epsilon`, `lr`, `lr_scan`, `seedvar` and `budget` were run (recall stages
  from `mqar/`). Their outputs are in `results/pilot/`.
* The stages `lm_small_d_lr` and `lm_small_d` (small query-key dimension and the uniform-attention control) were added after the pilot
  and have only had smoke tests. No results exist for them. The same holds for the per-head-size calibration and timing stages and for
  the head-size options of `bench` and `full`.
* The pilot numbers were produced in two environments that are both Linux: a cloud sandbox (2 vCPU) and the Linux VM of the desktop
  app (4 vCPU, i5-1038NG7). No run was made natively on macOS.
* Some early pilot stages used r = 1e-6 and the matrix-multiply path for p = 2. Later stages used r = 1e-12 and the direct kernel.
  Neither r = 1e-12 nor the direct kernel has been approved by the user.
* The training engine was split into a task-agnostic core and task plug-ins on 2026-10-05. Runs through the new engine are
  bit-identical to the old combined engine for the language model and for the recall task (weights, results and run ids).

## What may be worth reusing

These are pointers, not an instruction. Anything reused should be re-checked against the new specification.

* `msa/lp.py` and `msa/gradcheck.py`: the L^p kernel with a hand-written backward pass, tested against naive autograd in float64
  (worst error 7e-15), with the Chebyshev (p = infinity) tie rule. This is the piece with the most testing behind it.
* `msa/calibration.py`: the layer-by-layer calibration with the diagonal masked.
* The shared-initialisation rule in `msa/models.py` (a generator seeded by run seed and parameter name), which makes common parameters
  identical across arms.
* The bit-exact checkpoint and resume logic in `msa/train.py`.
* Cautions: `DESIGN.md` lists what was an unapproved implementation choice, and `PILOT_FINDINGS.md` lists what the pilot did not measure.
