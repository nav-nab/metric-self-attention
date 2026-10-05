"""The analysis of the main experiment: tests 1 to 5 of design section 9, tables and figures.

Only result files are read. The outcome is the final validation bpc of the main runs.
A diverged run has no outcome and enters the analysis as a missing value.
Test 6 is not coded. It is run only if the write-up claims an equivalence.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .data import FINAL_SEEDS
from .scores import HEAD_SIZES, P_VALUES, arm_name_for_p
from .sweep import all_cells, main_dir
from .util import atomic_write_text

LP_ARMS = tuple(arm_name_for_p(p) for p in P_VALUES)
INV_P = np.array([0.0 if math.isinf(p) else 1.0 / p for p in P_VALUES])    # the axis 1/p
LOG2_D = np.log2(np.array(HEAD_SIZES, dtype=float))
P_LABELS = ["∞" if math.isinf(p) else f"{p:g}" for p in P_VALUES]


class AnalysisError(RuntimeError):
    pass


# ---------------------------------------------------------------- loading

def load_main_results(root, seeds=FINAL_SEEDS):
    """All main runs as a table. A diverged run has the outcome NaN."""
    rows, missing = [], []
    for cell in all_cells():
        for seed in seeds:
            path = main_dir(root, cell, seed) / "result.json"
            if not path.exists():
                missing.append(path.parent.name)
                continue
            r = json.loads(path.read_text())
            bpc = r["final_val_bpc"]
            rows.append({"cell": cell.id, "arm": cell.arm, "D": cell.head_dim, "seed": seed, "lr": r["lr"],
                         "val_bpc": np.nan if (r["diverged"] or bpc is None) else bpc,
                         "diverged": bool(r["diverged"]), "diverged_reason": r.get("diverged_reason"),
                         "final_train_loss": r.get("final_train_loss")})
    if missing:
        raise AnalysisError(f"{len(missing)} main runs are missing, for example {missing[:3]}. "
                            "Run the main stage first.")
    return pd.DataFrame(rows)


class Grid:
    """The outcomes as arrays over the seeds: y(arm, D) has one value per seed, NaN if diverged."""

    def __init__(self, df, seeds=FINAL_SEEDS):
        self.seeds = list(seeds)
        self._y = {}
        for (arm, D), part in df.groupby(["arm", "D"]):
            part = part.set_index("seed").loc[self.seeds]
            self._y[(arm, int(D))] = part["val_bpc"].to_numpy(dtype=float)

    def y(self, arm, D):
        return self._y[(arm, D)]

    def lp_array(self):
        """The array (seed, head size, exponent) of the seven L^p arms."""
        return np.stack([np.stack([self.y(arm, D) for arm in LP_ARMS], axis=1) for D in HEAD_SIZES], axis=1)

    def gap(self, arm, D):
        return self.y(arm, D) - self.y("dot_tied", D)


# ---------------------------------------------------------------- small helpers

def one_sample(values):
    """Mean, 95% interval, t statistic and two-sided p-value of a one-sample t-test against 0."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    out = {"n": n, "mean": float(x.mean()) if n else math.nan, "ci_low": math.nan, "ci_high": math.nan,
           "t": math.nan, "p": math.nan}
    if n < 2:
        return out
    se = x.std(ddof=1) / math.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * se
    out.update(ci_low=out["mean"] - half, ci_high=out["mean"] + half)
    if se > 0:
        out["t"] = out["mean"] / se
        out["p"] = float(2 * stats.t.sf(abs(out["t"]), n - 1))
    return out


def holm(p_values):
    """Holm-adjusted p-values. NaN entries are left out of the family."""
    p = np.asarray(p_values, dtype=float)
    adj = np.full_like(p, np.nan)
    idx = np.where(~np.isnan(p))[0]
    order = idx[np.argsort(p[idx])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def slope_on_log2_d(values_by_d, min_points=3):
    """Least-squares slope of a quantity on log2 D, using the head sizes that have a value."""
    v = np.asarray(values_by_d, dtype=float)
    ok = ~np.isnan(v)
    if ok.sum() < min_points:
        return math.nan
    return float(np.polyfit(LOG2_D[ok], v[ok], 1)[0])


def _seed_columns(values, seeds):
    return {f"seed_{s}": v for s, v in zip(seeds, values)}


# ---------------------------------------------------------------- test 1

def rm_anova_interaction(Y):
    """The F-test of the D x p interaction in a repeated-measures design (plan section 8.1).

    Y has the shape (seeds, head sizes, exponents) and no missing values.
    """
    S, ND, NP = Y.shape
    m_sd = Y.mean(axis=2, keepdims=True)
    m_sp = Y.mean(axis=1, keepdims=True)
    m_dp = Y.mean(axis=0, keepdims=True)
    m_s = Y.mean(axis=(1, 2), keepdims=True)
    m_d = Y.mean(axis=(0, 2), keepdims=True)
    m_p = Y.mean(axis=(0, 1), keepdims=True)
    g = Y.mean()
    ss_int = S * ((m_dp - m_d - m_p + g) ** 2).sum()
    ss_err = ((Y - m_sd - m_sp - m_dp + m_s + m_d + m_p - g) ** 2).sum()
    df1 = (ND - 1) * (NP - 1)
    df2 = (S - 1) * df1
    F = (ss_int / df1) / (ss_err / df2)
    return {"F": float(F), "df1": df1, "df2": df2, "p": float(stats.f.sf(F, df1, df2)),
            "ss_interaction": float(ss_int), "ss_error": float(ss_err)}


def analyze_interaction(grid):
    """Test 1 with the rule for missing values: an exponent with any diverged run is removed."""
    Y = grid.lp_array()
    bad = np.isnan(Y).any(axis=(0, 1))
    removed = [LP_ARMS[i] for i in np.where(bad)[0]]
    keep = ~bad
    if keep.sum() < 2:
        return {"F": math.nan, "df1": math.nan, "df2": math.nan, "p": math.nan, "removed_arms": ";".join(removed)}
    out = rm_anova_interaction(Y[:, :, keep])
    out["removed_arms"] = ";".join(removed)
    return out


# ---------------------------------------------------------------- tests 2 and 3

def analyze_gaps(grid):
    """The paired gap of each L^p arm to the tied dot product at each head size."""
    rows = []
    for D in HEAD_SIZES:
        gaps = [grid.gap(arm, D) for arm in LP_ARMS]
        res = [one_sample(g) for g in gaps]
        adj = holm([r["p"] for r in res])
        for arm, g, r, a in zip(LP_ARMS, gaps, res, adj):
            rows.append({"D": D, "comparison": f"{arm} - dot_tied", "family": "Holm, seven arms at this D",
                         **r, "p_holm": a, **_seed_columns(g, grid.seeds)})
    for D in HEAD_SIZES:
        g = grid.gap("dot_untied", D)
        rows.append({"D": D, "comparison": "dot_untied - dot_tied", "family": "uncorrected", **one_sample(g),
                     "p_holm": math.nan, **_seed_columns(g, grid.seeds)})
    g = grid.y("dot_untied_unmasked", 16) - grid.y("dot_untied", 16)
    rows.append({"D": 16, "comparison": "dot_untied_unmasked - dot_untied", "family": "uncorrected",
                 **one_sample(g), "p_holm": math.nan, **_seed_columns(g, grid.seeds)})
    return pd.DataFrame(rows)


def analyze_trend(grid):
    """The trend of the gap with log2 D: one slope per seed, then a one-sample t-test over the seeds."""
    S = len(grid.seeds)
    gaps = {arm: np.stack([grid.gap(arm, D) for D in HEAD_SIZES], axis=1) for arm in LP_ARMS}   # (seed, D)
    mean_gap = np.mean([gaps[a] for a in LP_ARMS], axis=0)     # NaN if any arm is missing at that D
    rows = []
    for name, arr in list(gaps.items()) + [("mean of the seven arms", mean_gap)]:
        slopes = np.array([slope_on_log2_d(arr[s]) for s in range(S)])
        rows.append({"arm": name, **one_sample(slopes), **_seed_columns(slopes, grid.seeds)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- test 4

def contrasts(y):
    """The linear and quadratic contrasts of the seven values along 1/p.

    The linear contrast is the slope of the least-squares line on 1/p. The quadratic contrast is the
    coefficient of (1/p)^2 in the least-squares quadratic. Both are NaN if a value is missing.
    """
    y = np.asarray(y, dtype=float)
    if np.isnan(y).any():
        return math.nan, math.nan
    return float(np.polyfit(INV_P, y, 1)[0]), float(np.polyfit(INV_P, y, 2)[0])


def analyze_shape(grid):
    S = len(grid.seeds)
    lin = np.full((S, len(HEAD_SIZES)), np.nan)
    quad = np.full((S, len(HEAD_SIZES)), np.nan)
    for j, D in enumerate(HEAD_SIZES):
        Y = np.stack([grid.y(arm, D) for arm in LP_ARMS], axis=1)         # (seed, exponent)
        for s in range(S):
            lin[s, j], quad[s, j] = contrasts(Y[s])
    rows = []
    for name, arr in (("linear", lin), ("quadratic", quad)):
        for j, D in enumerate(HEAD_SIZES):
            rows.append({"contrast": name, "at": f"D = {D}", **one_sample(arr[:, j]), **_seed_columns(arr[:, j], grid.seeds)})
        slopes = np.array([slope_on_log2_d(arr[s]) for s in range(S)])
        rows.append({"contrast": name, "at": "slope on log2 D", **one_sample(slopes), **_seed_columns(slopes, grid.seeds)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- test 5

def bootstrap_resamples(S):
    """Every resample of S seeds with replacement, as counts per seed, with its exact probability."""
    counts = [c for c in itertools.product(range(S + 1), repeat=S) if sum(c) == S]
    weights = [math.factorial(S) / math.prod(math.factorial(x) for x in c) / S ** S for c in counts]
    return np.array(counts, dtype=float), np.array(weights)


def _weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights)[order]
    cum = np.cumsum(w) - 0.5 * w
    return float(np.interp(q, cum / w.sum(), v))


def analyze_best(grid):
    """Bootstrap over whole seeds, computed exactly over all 126 distinct resamples."""
    Y = grid.lp_array()                                                     # (seed, D, p)
    S, ND, NP = Y.shape
    counts, weights = bootstrap_resamples(S)
    valid = ~np.isnan(Y)
    filled = np.nan_to_num(Y)
    best = np.zeros((ND, NP))
    pair_corr = np.full(len(counts), np.nan)
    pairs = list(itertools.combinations(range(ND), 2))
    for i, (c, w) in enumerate(zip(counts, weights)):
        num = np.einsum("s,sdp->dp", c, filled)
        den = np.einsum("s,sdp->dp", c, valid.astype(float))
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = num / den
        for d in range(ND):
            if not np.isnan(mean[d]).any():
                best[d, np.argmin(mean[d])] += w
        cors = [stats.spearmanr(mean[a], mean[b])[0] for a, b in pairs if not (np.isnan(mean[a]).any() or np.isnan(mean[b]).any())]
        if cors:
            pair_corr[i] = float(np.mean(cors))
    full = np.nanmean(Y, axis=0)                                            # (D, p)
    corr_matrix = np.array([[stats.spearmanr(full[a], full[b])[0] for b in range(ND)] for a in range(ND)])
    ok = ~np.isnan(pair_corr)
    summary = {
        "mean_pairwise_rank_correlation": float(np.average(pair_corr[ok], weights=weights[ok])) if ok.any() else math.nan,
        "ci_low": _weighted_quantile(pair_corr[ok], weights[ok], 0.025) if ok.any() else math.nan,
        "ci_high": _weighted_quantile(pair_corr[ok], weights[ok], 0.975) if ok.any() else math.nan,
        "n_resamples": len(counts),
    }
    return best, corr_matrix, summary


# ---------------------------------------------------------------- tables

def cell_means_table(df):
    rows = []
    for cell in all_cells():
        part = df[df["cell"] == cell.id]
        y = part["val_bpc"].to_numpy(dtype=float)
        ok = y[~np.isnan(y)]
        rows.append({"arm": cell.arm, "D": "-" if cell.arm == "uniform" else cell.head_dim, "lr": part["lr"].iloc[0],
                     "n_valid": len(ok), "mean_bpc": ok.mean() if len(ok) else math.nan,
                     "sd_bpc": ok.std(ddof=1) if len(ok) > 1 else math.nan,
                     "min_bpc": ok.min() if len(ok) else math.nan, "max_bpc": ok.max() if len(ok) else math.nan})
    return pd.DataFrame(rows)


def chosen_rates_table(root):
    path = Path(root) / "selected_rates.json"
    if not path.exists():
        return None
    sel = json.loads(path.read_text())
    rates = ["0.001", "0.003", "0.01", "0.03", "0.09"]
    rows = []
    for cell in all_cells():
        item = sel["cells"][cell.id]
        row = {"arm": cell.arm, "D": "-" if cell.arm == "uniform" else cell.head_dim, "chosen_lr": item["lr"]}
        for r in rates:
            scan = item["scan"].get(r)
            row[f"bpc_at_{r}"] = "" if scan is None else ("diverged" if scan["diverged"] else scan["val_bpc"])
        rows.append(row)
    return pd.DataFrame(rows)


def _write_csv(df, path):
    atomic_write_text(path, df.to_csv(index=False, float_format="%.6g"))


# ---------------------------------------------------------------- figures

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
# one-hue ordinal ramp for the ordered exponents (steps 250 to 650 of the blue ramp)
P_RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#104281"]
D_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
                         "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "regular",
                         "legend.frameon": False})
    return plt


def _d_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(HEAD_SIZES)
    ax.set_xticklabels([str(d) for d in HEAD_SIZES])
    ax.minorticks_off()
    ax.set_xlim(1.7, 18.5)


def fig_bpc_by_head_size(grid, path):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for arm, label, color in zip(LP_ARMS, P_LABELS, P_RAMP):
        ax.plot(HEAD_SIZES, [np.nanmean(grid.y(arm, D)) for D in HEAD_SIZES], "-o", color=color, lw=2, ms=5,
                label=f"L^p, p = {label}")
    ax.plot(HEAD_SIZES, [np.nanmean(grid.y("dot_tied", D)) for D in HEAD_SIZES], "-o", color=INK, lw=2.4, ms=5,
            label="tied dot product")
    ax.plot(HEAD_SIZES, [np.nanmean(grid.y("dot_untied", D)) for D in HEAD_SIZES], "--s", color=ORANGE, lw=2, ms=5,
            label="untied dot product")
    ax.axhline(np.nanmean(grid.y("uniform", 16)), color=MUTED, lw=1.6, ls=":", label="uniform attention")
    _d_axis(ax)
    ax.set_xlabel("head size D")
    ax.set_ylabel("validation bits per character (mean of seeds)")
    ax.set_title("Validation bpc by head size", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_gap_by_head_size(grid, path):
    plt = _pyplot()
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.4), sharex=True, sharey=True)
    series = [(f"p = {lab}", arm) for lab, arm in zip(P_LABELS, LP_ARMS)] + [("mean of the seven arms", None)]
    for ax, (title, arm) in zip(axes.ravel(), series):
        if arm is None:
            gaps = np.stack([np.mean([grid.gap(a, D) for a in LP_ARMS], axis=0) for D in HEAD_SIZES])   # (D, seed)
        else:
            gaps = np.stack([grid.gap(arm, D) for D in HEAD_SIZES])
        for k in range(gaps.shape[1]):
            ax.plot(HEAD_SIZES, gaps[:, k], "o", color=MUTED, ms=3.5, alpha=0.55, lw=0)
        ax.plot(HEAD_SIZES, np.nanmean(gaps, axis=1), "-o", color=BLUE, lw=2, ms=5)
        ax.axhline(0, color=INK2, lw=1)
        _d_axis(ax)
        ax.set_title(title, loc="left")
    for ax in axes[1]:
        ax.set_xlabel("head size D")
    for ax in axes[:, 0]:
        ax.set_ylabel("gap to tied dot (bpc)")
    fig.suptitle("Paired gap to the tied dot product; negative means the distance is better. Grey dots are single seeds.",
                 x=0.01, ha="left", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_shape_over_p(grid, path):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for D, color in zip(HEAD_SIZES, D_RAMP):
        m = np.array([np.nanmean(grid.y(arm, D)) for arm in LP_ARMS])
        ax.plot(INV_P, m - m.mean(), "-o", color=color, lw=2, ms=5, label=f"D = {D}")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(INV_P)
    ax.set_xticklabels(P_LABELS)
    ax.invert_xaxis()
    ax.set_xlabel("exponent p (the axis is 1/p)")
    ax.set_ylabel("bpc minus the mean over p at the same D")
    ax.set_title("Shape of the effect of p at each head size", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_best_probability(best, path):
    plt = _pyplot()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_RAMP)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    im = ax.imshow(best, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(P_LABELS)))
    ax.set_xticklabels(P_LABELS)
    ax.set_yticks(range(len(HEAD_SIZES)))
    ax.set_yticklabels([str(d) for d in HEAD_SIZES])
    for i in range(best.shape[0]):
        for j in range(best.shape[1]):
            ax.text(j, i, f"{best[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if best[i, j] > 0.5 else INK)
    ax.set_xlabel("exponent p")
    ax.set_ylabel("head size D")
    ax.set_title("Probability that each p has the lowest mean bpc (bootstrap over seeds)", loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- the stage

def run_analysis(root, out_dir=None, log=print):
    root = Path(root)
    out = Path(out_dir) if out_dir is not None else root / "analysis"
    tables, figures = out / "tables", out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    df = load_main_results(root)
    grid = Grid(df)

    _write_csv(df, tables / "results_long.csv")
    _write_csv(cell_means_table(df), tables / "cell_means.csv")
    _write_csv(df[df["diverged"]][["cell", "arm", "D", "seed", "lr", "diverged_reason"]], tables / "diverged_runs.csv")
    chosen = chosen_rates_table(root)
    if chosen is not None:
        _write_csv(chosen, tables / "chosen_rates.csv")

    t1 = analyze_interaction(grid)
    _write_csv(pd.DataFrame([t1]), tables / "test1_interaction.csv")
    _write_csv(analyze_gaps(grid), tables / "test2_gaps.csv")
    t3 = analyze_trend(grid)
    _write_csv(t3, tables / "test3_trend.csv")
    _write_csv(analyze_shape(grid), tables / "test4_shape.csv")
    best, corr, summary = analyze_best(grid)
    _write_csv(pd.DataFrame(best, index=HEAD_SIZES, columns=P_LABELS).rename_axis("D").reset_index(),
               tables / "test5_best_probability.csv")
    _write_csv(pd.DataFrame(corr, index=HEAD_SIZES, columns=[f"D={d}" for d in HEAD_SIZES]).rename_axis("D").reset_index(),
               tables / "test5_rank_correlation.csv")
    _write_csv(pd.DataFrame([summary]), tables / "test5_rank_stability.csv")

    fig_bpc_by_head_size(grid, figures / "bpc_by_head_size.png")
    fig_gap_by_head_size(grid, figures / "gap_by_head_size.png")
    fig_shape_over_p(grid, figures / "shape_over_p.png")
    fig_best_probability(best, figures / "best_probability.png")

    n_div = int(df["diverged"].sum())
    log(f"analysis: {len(df)} main runs read, {n_div} diverged")
    if math.isnan(t1["F"]):
        log("analysis: test 1 could not be computed (fewer than two exponents without divergence)")
    else:
        log(f"analysis: test 1, D x p interaction: F({t1['df1']}, {t1['df2']}) = {t1['F']:.3f}, p = {t1['p']:.4g}"
            + (f" (removed: {t1['removed_arms']})" if t1["removed_arms"] else ""))
    row = t3.iloc[-1]
    log(f"analysis: test 3, mean gap slope per doubling of D = {row['mean']:.4f} bpc, p = {row['p']:.4g}")
    log(f"analysis: tables and figures are in {out}")
    return {"test1": t1, "test3": t3, "test5": summary, "out_dir": out}


# ---------------------------------------------------------------- invented results (smoke and tests)

def write_invented_results(root, seed=0, interaction=True):
    """Write invented main results for all 47 cells and five seeds, with a known pattern.

    The tied dot product improves slowly with D. Each L^p arm has a gap that is most negative at small D
    and fades with D. The shape over p changes with D. Seeds share an offset, so the data are paired.
    """
    rng = np.random.default_rng(seed)
    root = Path(root)
    offset = rng.normal(0, 0.02, len(FINAL_SEEDS))

    def bpc(arm, D, s):
        tied = 2.75 - 0.05 * math.log2(D)
        if arm == "dot_tied":
            base = tied
        elif arm == "dot_untied":
            base = tied - 0.01
        elif arm == "dot_untied_unmasked":
            base = tied - 0.03
        elif arm == "uniform":
            base = 3.3
        else:
            x = 0.0 if arm.endswith("inf") else 1.0 / float(arm[3:])
            scale = 0.10 * (2.0 / D) ** 0.5
            base = tied - scale * (1.0 - (x - 0.4) ** 2 * (1.5 if interaction else 0.0) * (4.0 / D))
        return base + offset[s] + rng.normal(0, 0.01)

    sel = {"cells": {}, "needs_extra": [], "incomplete": [], "all_diverged": [], "complete": True}
    for cell in all_cells():
        for s_index, s in enumerate(FINAL_SEEDS):
            d = main_dir(root, cell, s)
            d.mkdir(parents=True, exist_ok=True)
            (d / "result.json").write_text(json.dumps({
                "arm": cell.arm, "D": cell.head_dim, "seed": s, "stage": "main", "lr": 0.01, "steps": 4000,
                "final_val_bpc": bpc(cell.arm, cell.head_dim, s_index), "final_train_loss": 2.2,
                "diverged": False, "diverged_reason": None, "steps_done": 4000, "wall_time_s": 1.0, "resumes": 0}))
        sel["cells"][cell.id] = {"arm": cell.arm, "D": cell.head_dim, "lr": 0.01, "val_bpc": 2.5,
                                 "scan": {"0.003": {"val_bpc": 2.52, "diverged": False},
                                          "0.01": {"val_bpc": 2.5, "diverged": False},
                                          "0.03": {"val_bpc": 2.6, "diverged": False}}}
    (root / "selected_rates.json").write_text(json.dumps(sel))
