# Distance attention at small head sizes for a character-level language model

This repository replaces the query-key dot product in self-attention by a negative $\ell^p$ distance score and measures the effect on a small language model.
The study varies the exponent $p$ in $\{1, 1.5, 2, 3, 4, 8, \infty\}$ and the head size $D$ (the length of the query and key vectors) in $\{2, 4, 8, 12, 16\}$.
The model is a two-layer, character-level transformer trained on Tiny Shakespeare, and the outcome is the final validation bits per character.
Each of the 47 cells is run with five seeds, which gives 235 main runs of 4,000 steps.

The paper in `paper/main.pdf` reports the design, the results and the statistical analysis.
In this setting the tied dot product scores better than every $\ell^p$ variant in 34 of 35 cells, and the gap shrinks as $D$ grows.

## Layout

| Path | Content |
|---|---|
| `paper/` | The paper (`main.tex`, `main.pdf`), its figures and tables, the exported data, and the scripts that build them. |
| `docs/` | The design record: `project_outline.md`, `design_details.md`, `implementation_plan.md`, `results_guide.md` and `analysis_findings.md`. Read them in this order. |
| `msa/` | The code: scores, model, training, calibration, sweep and analysis. |
| `tests/` | The unit tests. |
| `execute_project.py` | The single entry point. |
| `results/` | Calibration constants, the learning-rate scan, the main runs with their final weights, and the analysis tables and figures. |
| `data/` | The corpus, `tinyshakespeare.txt` (1,115,394 characters, md5 `6fb458f1232090904fb40fe944165e91`). |

## Reproducing the results

The pinned environment is in `requirements.txt` (torch 2.2.2, numpy below 2).
The stages run in this order, and finished runs are skipped when a stage is repeated.

```
python execute_project.py check       # unit tests
python execute_project.py all         # calibrate, scan, select, main, analyze
python execute_project.py analyze     # tables, tests and figures from the existing results
```

The full run took about two days on a laptop CPU.
The `results/` folder already holds its output, so `analyze` alone reproduces the tables and figures in `results/analysis/`.

## Rebuilding the paper

```
python paper/export_data.py     # copies the analysis tables into paper/data/
python paper/build_assets.py    # builds the paper's figures and tables
cd paper && pdflatex main.tex && pdflatex main.tex
```

`paper/verify_tests.py` recomputes the paired tests independently of `msa/analyze.py`, and `paper/verify_tests_output.txt` holds its output.

## Notes

The design, the code, the analysis and the paper were developed in collaboration with Claude (Anthropic), as the paper's acknowledgements state.

## Licence

MIT. See `LICENSE`.
