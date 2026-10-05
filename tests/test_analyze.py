import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from msa import analyze
from msa.analyze import (AnalysisError, Grid, LP_ARMS, analyze_best, analyze_interaction, analyze_shape,
                         analyze_trend, bootstrap_resamples, contrasts, holm, load_main_results, one_sample,
                         rm_anova_interaction, run_analysis, slope_on_log2_d, write_invented_results)
from msa.scores import HEAD_SIZES
from msa.sweep import Cell, main_dir


def test_anova_agrees_with_statsmodels_anovarm():
    from statsmodels.stats.anova import AnovaRM
    rng = np.random.default_rng(1)
    Y = rng.normal(0, 0.02, (5, 5, 7)) + rng.normal(0, 0.03, (5, 1, 1)) + 0.01 * np.arange(7)[None, None, :]
    Y[:, 2, 3] += 0.02                                       # some structure
    ours = rm_anova_interaction(Y)
    rows = [(s, d, p, Y[s, d, p]) for s in range(5) for d in range(5) for p in range(7)]
    long = pd.DataFrame(rows, columns=["s", "d", "p", "y"])
    table = AnovaRM(long, "y", "s", within=["d", "p"]).fit().anova_table
    assert ours["df1"] == 24 and ours["df2"] == 96
    assert ours["F"] == pytest.approx(table.loc["d:p", "F Value"], rel=1e-9)
    assert ours["p"] == pytest.approx(table.loc["d:p", "Pr > F"], rel=1e-6)


def test_anova_detects_an_interaction_and_not_a_null():
    rng = np.random.default_rng(2)
    seed_level = rng.normal(0, 0.03, (5, 1, 1))
    main_effects = 0.1 * np.arange(5)[None, :, None] + 0.05 * np.arange(7)[None, None, :]
    noise = rng.normal(0, 0.01, (5, 5, 7))
    null = rm_anova_interaction(seed_level + main_effects + noise)
    assert null["p"] > 0.01
    inter = 0.03 * np.outer(np.arange(5) - 2, np.arange(7) - 3)[None]
    assert rm_anova_interaction(seed_level + main_effects + inter + noise)["p"] < 1e-6


def grid_from_array(Y):
    """A Grid built from an array (seed, D, exponent) for the L^p arms, with dummy dot arms."""
    rows = []
    for s in range(Y.shape[0]):
        for d, D in enumerate(HEAD_SIZES):
            for p, arm in enumerate(LP_ARMS):
                rows.append({"arm": arm, "D": D, "seed": s + 1, "val_bpc": Y[s, d, p]})
            for arm in ("dot_tied", "dot_untied", "dot_untied_unmasked", "uniform"):
                rows.append({"arm": arm, "D": D if arm != "uniform" else 16, "seed": s + 1, "val_bpc": 2.5 + 0.001 * s})
    df = pd.DataFrame(rows).drop_duplicates(["arm", "D", "seed"])
    return Grid(df, seeds=list(range(1, Y.shape[0] + 1)))


def test_a_removed_exponent_changes_the_degrees_of_freedom():
    rng = np.random.default_rng(3)
    Y = 2.5 + rng.normal(0, 0.02, (5, 5, 7))
    full = analyze_interaction(grid_from_array(Y))
    assert (full["df1"], full["df2"]) == (24, 96) and full["removed_arms"] == ""
    Y[2, 1, 5] = np.nan                                      # one diverged run of lp_8
    cut = analyze_interaction(grid_from_array(Y))
    assert cut["removed_arms"] == "lp_8" and (cut["df1"], cut["df2"]) == (20, 80)
    Y[:, :, :6] = np.nan
    assert math.isnan(analyze_interaction(grid_from_array(Y))["F"])


def test_holm_adjustment():
    adj = holm([0.01, 0.04, 0.03, 0.5])
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert holm([0.2, np.nan, 0.01])[1] != holm([0.2, np.nan, 0.01])[1]      # NaN stays NaN
    assert holm([0.2, np.nan, 0.01])[2] == pytest.approx(0.02)


def test_one_sample_matches_scipy():
    x = [0.1, -0.02, 0.07, 0.03, 0.05]
    r = one_sample(x)
    t = stats.ttest_1samp(x, 0.0)
    assert r["t"] == pytest.approx(t.statistic) and r["p"] == pytest.approx(t.pvalue)
    ci = t.confidence_interval(0.95)
    assert (r["ci_low"], r["ci_high"]) == pytest.approx((ci.low, ci.high))
    assert one_sample([1.0, np.nan])["n"] == 1 and math.isnan(one_sample([1.0, np.nan])["p"])


def test_trend_test_recovers_a_known_slope_per_doubling():
    Y = np.zeros((5, 5, 7))
    Y[:] = 2.5
    for d, D in enumerate(HEAD_SIZES):
        Y[:, d, :] += 0.02 * np.log2(D)                      # the L^p arms rise by 0.02 per doubling of D
    grid = grid_from_array(Y)                                # the tied dot product is flat (2.5 + tiny seed shifts)
    t = analyze_trend(grid)
    assert len(t) == 8 and t.iloc[-1]["arm"] == "mean of the seven arms"
    assert t["mean"].to_numpy() == pytest.approx([0.02] * 8, abs=1e-9)
    assert slope_on_log2_d([1, 2, 3, np.nan, np.nan]) != slope_on_log2_d([1, 2, 3, 4, 5])
    assert math.isnan(slope_on_log2_d([1, 2, np.nan, np.nan, np.nan]))


def test_contrasts_recover_a_known_quadratic():
    from msa.analyze import INV_P
    y = 2.0 + 0.3 * INV_P + 0.5 * INV_P ** 2
    lin, quad = contrasts(y)
    assert quad == pytest.approx(0.5)
    assert lin == pytest.approx(np.polyfit(INV_P, y, 1)[0])
    assert all(math.isnan(v) for v in contrasts([1, 2, np.nan, 4, 5, 6, 7]))
    # a pure line has no quadratic part
    assert contrasts(1.0 + 2.0 * INV_P) == pytest.approx((2.0, 0.0))


def test_test4_table_layout():
    rng = np.random.default_rng(4)
    t = analyze_shape(grid_from_array(2.5 + rng.normal(0, 0.02, (5, 5, 7))))
    assert len(t) == 2 * (5 + 1) and set(t["contrast"]) == {"linear", "quadratic"}
    assert t["at"].tolist()[5] == "slope on log2 D"


def test_bootstrap_has_126_exact_resamples():
    counts, weights = bootstrap_resamples(5)
    assert len(counts) == 126 and weights.sum() == pytest.approx(1.0)
    assert (counts.sum(axis=1) == 5).all()
    assert weights[(counts == 5).any(axis=1)].sum() == pytest.approx(5 / 5 ** 5)     # all five draws equal


def test_probability_of_being_best():
    rng = np.random.default_rng(5)
    Y = 2.5 + rng.normal(0, 0.001, (5, 5, 7))
    Y[:, :, 4] -= 0.5                                        # p = 4 is clearly best everywhere
    best, corr, summary = analyze_best(grid_from_array(Y))
    assert best[:, 4] == pytest.approx(1.0) and best.sum(axis=1) == pytest.approx(1.0)
    assert corr.shape == (5, 5) and summary["n_resamples"] == 126
    # each seed prefers a different exponent, so no exponent is clearly best
    Y2 = np.full((5, 5, 7), 2.6) + 1e-6 * np.arange(7)
    for s in range(5):
        Y2[s, :, s] = 2.5
    best2, _, _ = analyze_best(grid_from_array(Y2))
    assert best2[:, :5].max() < 0.5 and best2[:, 5:].sum() == pytest.approx(0.0)
    assert best2.sum(axis=1) == pytest.approx(1.0)


def test_loading_requires_every_main_run(tmp_path):
    with pytest.raises(AnalysisError, match="main runs are missing"):
        load_main_results(tmp_path)
    write_invented_results(tmp_path)
    assert len(load_main_results(tmp_path)) == 235
    (main_dir(tmp_path, Cell("lp_2", 8), 3) / "result.json").unlink()
    with pytest.raises(AnalysisError, match="1 main runs are missing"):
        load_main_results(tmp_path)


def test_the_analysis_stage_on_invented_results(tmp_path):
    write_invented_results(tmp_path)
    messages = []
    out = run_analysis(tmp_path, log=messages.append)
    tables = tmp_path / "analysis" / "tables"
    figures = tmp_path / "analysis" / "figures"
    for name in ["results_long", "cell_means", "diverged_runs", "chosen_rates", "test1_interaction", "test2_gaps",
                 "test3_trend", "test4_shape", "test5_best_probability", "test5_rank_correlation",
                 "test5_rank_stability"]:
        assert (tables / f"{name}.csv").exists(), name
    for name in ["bpc_by_head_size", "gap_by_head_size", "shape_over_p", "best_probability"]:
        assert (figures / f"{name}.png").stat().st_size > 5000, name
    # the invented data have a known interaction, a negative gap that fades with D, and no divergence
    assert out["test1"]["p"] < 1e-6 and (out["test1"]["df1"], out["test1"]["df2"]) == (24, 96)
    mean_row = out["test3"].iloc[-1]
    assert mean_row["mean"] > 0 and mean_row["p"] < 0.01
    gaps = pd.read_csv(tables / "test2_gaps.csv")
    assert len(gaps) == 35 + 5 + 1 and gaps[gaps["family"].str.startswith("Holm")]["mean"].mean() < 0
    assert len(pd.read_csv(tables / "diverged_runs.csv")) == 0
    assert any("test 1" in m for m in messages)


def test_a_diverged_run_is_a_missing_value_and_is_reported(tmp_path):
    write_invented_results(tmp_path)
    path = main_dir(tmp_path, Cell("lp_8", 4), 2) / "result.json"
    r = json.loads(path.read_text())
    r.update(diverged=True, final_val_bpc=None, diverged_reason="non-finite training loss")
    path.write_text(json.dumps(r))
    out = run_analysis(tmp_path, log=lambda m: None)
    assert out["test1"]["removed_arms"] == "lp_8" and out["test1"]["df1"] == 20
    div = pd.read_csv(tmp_path / "analysis" / "tables" / "diverged_runs.csv")
    assert div["cell"].tolist() == ["lp_8_D4"] and div["seed"].tolist() == [2]
    gaps = pd.read_csv(tmp_path / "analysis" / "tables" / "test2_gaps.csv")
    row = gaps[(gaps["comparison"] == "lp_8 - dot_tied") & (gaps["D"] == 4)].iloc[0]
    assert row["n"] == 4 and pd.isna(row["seed_2"])
