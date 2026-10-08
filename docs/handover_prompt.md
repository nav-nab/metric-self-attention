# Handover prompt for the analysing agent

You are taking over the Metric Self-Attention project at the analysis stage. The design is settled, the code is written and tested, and the full run has finished. Your task is to analyse the results with the user and to discuss them. You do not change the design and you do not run any training.

## The project

The project tests whether replacing the query-key dot product in self-attention by a negative L^p distance helps, and how the answer depends on the head size D. The exponents are p in {1, 1.5, 2, 3, 4, 8, infinity}. The setting is a small character-level language model on Tiny Shakespeare. The grid has 47 cells: nine arms at D in {2, 4, 8, 12, 16}, plus a uniform-attention control and an unmasked untied dot-product reference at D = 16. Each cell has five seeds. The outcome is the final validation bits per character.

## Reading order

Read these in the repository root (the folder metric-self-attention) before replying:

1. README.md
2. docs/project_outline.md
3. docs/design_details.md
4. docs/implementation_plan.md (section 8.1 for the test-1 form, section 13 for the coding decisions, section 14 for the record of the run)
5. docs/results_guide.md (file locations and schemas, the analysis definitions, points to keep in mind, write-up requirements, open items)
6. The project memory guide areas/metric-self-attention.md, then index.md

If a memory note and a repository document disagree, the document is current.

## Rules

- Ignore the shelved/ folder. Ignore the memory file areas/metric-self-attention-code.md.
- Do not train, scan or pilot. Do not edit anything under results/ except results/analysis/. You may rerun `python execute_project.py analyze` freely.
- The design, including the statistical plan, is fixed. If you think a change is needed, say so and ask the user.
- Commit locally only if the user asks. Never push. Never use `git add -A`.
- Write in formal prose with short, self-contained clauses. Add no extra tests, caveats or options unless they change a conclusion.
- Talk options through carefully and in detail. Ask the user when anything is unclear.
- Record user decisions in the memory guide as [stated] lines only after the user confirms them.

## Facts to keep in mind

- No run diverged. The rate scan chose 3e-3 in 46 of 47 cells and 1e-2 for the uniform control, so per-arm rate tuning is weak.
- The user has decided that no finer learning-rate scan is made. All conclusions rest on the existing run.
- The final weights are committed. The rest of results/ is untracked.
- The median seed sd is about 0.0096 bpc.
- Test 6 (equivalence) is not coded. Its margin is deferred and is needed only if an equivalence claim is made.
- The write-up must make the statements listed in design_details section 10, including that the baseline is non-standard because of the diagonal mask and that rates were tuned per arm on the reporting set.

## Your first reply

Do not analyse yet. Reply with:

1. A short summary of your understanding of the project, the design and the run.
2. The integrity checks you will make on results/ (for example that all 235 main runs are present, that each result.json agrees with its log.csv, and that a final.pt reloads).
3. Your questions for the user, including what they want from the discussion.

## If you cannot reach the user's files

If the device bridge is unavailable, ask the user to attach these files: everything in results/analysis/tables/, results/scan_table.csv, results/selected_rates.json and results/calibration/calibration_summary.csv.
