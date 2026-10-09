"""Draws the schematic figures used in docs/design_details.md.

No training is involved. Every figure is computed from formulas or from the
design tables in the document. Run:  python docs/figures/make_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d2"
BLUE, ORANGE, GREEN, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
GREY = "#b9b8b0"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- figure 1
def fig_grid():
    D = [2, 4, 8, 12, 16]
    rows = [("tied dot product (control)", ORANGE), (r"$\ell^{1}$", BLUE), (r"$\ell^{1.5}$", BLUE),
            (r"$\ell^{2}$", BLUE), (r"$\ell^{3}$", BLUE), (r"$\ell^{4}$", BLUE), (r"$\ell^{8}$", BLUE),
            (r"$\ell^{\infty}$", BLUE), ("untied dot product (reference)", ORANGE),
            ("uniform control", GREY), ("unmasked untied dot product", VIOLET)]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    n = len(rows)
    for r, (name, col) in enumerate(rows):
        y = n - 1 - r
        ax.text(-0.15, y + 0.5, name, ha="right", va="center", fontsize=10)
        if name == "uniform control":
            ax.add_patch(Rectangle((0.05, y + 0.08), 4.9, 0.84, fc=col, ec="white", lw=2))
            ax.text(2.5, y + 0.5, "one cell, the same for every D", ha="center",
                    va="center", fontsize=9, color=INK)
        elif name.startswith("unmasked"):
            ax.add_patch(Rectangle((4.05, y + 0.08), 0.9, 0.84, fc=col, ec="white", lw=2))
            ax.text(4.5, y + 0.5, "D = 16 only", ha="center", va="center",
                    fontsize=8.5, color="white")
        else:
            for c in range(5):
                ax.add_patch(Rectangle((c + 0.05, y + 0.08), 0.9, 0.84, fc=col,
                                       ec="white", lw=2))
    for c, d in enumerate(D):
        ax.text(c + 0.5, n + 0.25, f"D = {d}", ha="center", va="bottom", fontsize=10)
    ax.plot([0, 5], [n + 0.15, n + 0.15], color=INK2, lw=0.8)
    # group brackets on the right
    def bracket(y0, y1, label):
        ax.plot([5.15, 5.2, 5.2, 5.15], [y0, y0, y1, y1], color=INK2, lw=0.9)
        ax.text(5.3, (y0 + y1) / 2, label, va="center", fontsize=9, color=INK2)
    bracket(n - 8 + 0.1, n - 0.1, "crossed design:\n9 arms x 5 head sizes\n= 45 cells")
    bracket(1.1, 1.9, "extra: 1 cell")
    bracket(0.1, 0.9, "extra: 1 cell")
    ax.set_xlim(-0.2, 7.2)
    ax.set_ylim(-0.1, n + 0.7)
    ax.axis("off")
    ax.text(-0.15, -0.9, "Each cell is run for 5 final seeds, plus a one-seed learning-rate scan. "
            "45 + 2 = 47 cells.", fontsize=9, color=INK2, ha="left")
    ax.set_title("The experimental grid", loc="left", fontsize=12, pad=16)
    save(fig, "fig1_grid.png")


# ---------------------------------------------------------------- figure 2
def fig_geometry():
    fig, axes = plt.subplots(1, 5, figsize=(12.5, 2.9))
    lim = 2.0
    g = np.linspace(-lim, lim, 400)
    X, Y = np.meshgrid(g, g)

    def dist(p):
        if p == np.inf:
            return np.maximum(abs(X), abs(Y))
        return (abs(X) ** p + abs(Y) ** p) ** (1 / p)
    ax = axes[0]
    q = np.array([1.0, 0.45])
    Z = X * q[0] + Y * q[1]
    ax.contour(X, Y, Z, levels=np.arange(-3, 3.1, 0.75), colors=ORANGE, linewidths=1,
               negative_linestyles="solid")
    ax.annotate("", xy=(q[0] * 1.2, q[1] * 1.2), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.text(0.15, 0.95, "query", fontsize=8.5, color=INK)
    ax.set_title("dot product\nlevel sets are lines", fontsize=10)
    for ax, p, t in zip(axes[1:], [1, 2, 8, np.inf], ["1", "2", "8", "infinity"]):
        ax.contour(X, Y, dist(p), levels=[0.5, 1.0, 1.5, 2.0], colors=BLUE, linewidths=1)
        ax.plot([0], [0], "o", color=INK, ms=4)
        ax.text(0.08, 0.1, "query", fontsize=8.5, color=INK)
        ax.set_title(f"$\\ell^p$ distance, p = {t}\nlevel sets are balls", fontsize=10)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
    fig.suptitle("Level sets of the score of one query, in two dimensions (D = 2). "
                 "Each curve joins keys that get the same score.", x=0.01, ha="left",
                 fontsize=10, color=INK2, y=1.0)
    save(fig, "fig2_geometry.png")


# ---------------------------------------------------------------- figure 3
def fig_mask():
    T = 16
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.2),
                                  gridspec_kw={"width_ratios": [3, 1.6]})
    for i in range(T):
        for j in range(T):
            if j > i:
                fc = "white"
            elif i == j and i == 0:
                fc = ORANGE
            elif i == j:
                fc = GREY
            else:
                fc = BLUE
            ax.add_patch(Rectangle((j, T - 1 - i), 1, 1, fc=fc, ec="white", lw=1.2))
            if fc == "white":
                ax.add_patch(Rectangle((j, T - 1 - i), 1, 1, fc="white", ec=GRID, lw=0.4))
    ax.set_xlim(0, T)
    ax.set_ylim(0, T)
    ax.set_aspect("equal")
    ax.set_xlabel("key position j")
    ax.set_ylabel("query position i")
    ax.set_xticks([0.5, 4.5, 8.5, 12.5])
    ax.set_xticklabels(["0", "4", "8", "12"])
    ax.set_yticks([T - 0.5, T - 4.5, T - 8.5, T - 12.5])
    ax.set_yticklabels(["0", "4", "8", "12"])
    ax.set_title("Which keys each query may use (first 16 of 64 positions)",
                 fontsize=10, loc="left")
    ax.legend(handles=[Rectangle((0, 0), 1, 1, fc=BLUE, label="allowed"),
                       Rectangle((0, 0), 1, 1, fc=GREY, label="diagonal, masked"),
                       Rectangle((0, 0), 1, 1, fc=ORANGE, label="position 0 may use itself"),
                       Rectangle((0, 0), 1, 1, fc="white", ec=GRID, label="future, masked")],
              loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8.5,
              frameon=False)
    n = np.array([1] + list(range(1, T)))
    cols = [GREY if v < 8 else BLUE for v in n]
    ax2.barh(np.arange(T)[::-1], n, color=cols, height=0.8)
    ax2.axvline(8, color=INK2, lw=1, ls="--")
    ax2.set_ylim(-0.5, T - 0.5)
    ax2.set_yticks([])
    ax2.set_xlabel("allowed keys for query i")
    ax2.text(8.2, T - 1.6, "8", fontsize=9, color=INK2)
    ax2.set_title("Calibration uses only\nblue positions", fontsize=10, loc="left")
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)
    save(fig, "fig3_mask.png")


# ---------------------------------------------------------------- figure 4
def fig_floor():
    r = 1e-12
    p = np.array([1, 1.5, 2, 3, 4, 8])
    pc = np.linspace(1, 8, 200)
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.plot(pc, 100 * r ** (1 / pc), color=BLUE, lw=1.6)
    ax.plot(p, 100 * r ** (1 / p), "o", color=BLUE, ms=7, mec="white", mew=1.5)
    labels = {1: r"$10^{-10}$ %", 1.5: r"$10^{-6}$ %", 2: r"$10^{-4}$ %", 3: "0.01 %", 4: "0.1 %", 8: "3 %"}
    for x in p:
        v = 100 * r ** (1 / x)
        ax.annotate(labels[x], (x, v), textcoords="offset points",
                    xytext=(14 if x == 1 else 0, 9), ha="center", fontsize=8.5, color=INK)
    ax.set_yscale("log")
    ax.set_xlabel("p")
    ax.set_ylabel("floor on the distance,\npercent of the typical distance")
    ax.set_title("Floor created by the relative epsilon, r = 1e-12:  r^(1/p)", loc="left",
                 fontsize=10.5)
    ax.text(8.0, 1.2e-9, "p = infinity has no epsilon\nand no floor", ha="right",
            fontsize=8.5, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_ylim(1e-11, 40)
    save(fig, "fig4_floor.png")


# ---------------------------------------------------------------- figure 5
def fig_protocol():
    fig, ax = plt.subplots(figsize=(11, 4.1))
    ax.set_xlim(0, 106)
    ax.set_ylim(0, 42)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f3f2ee", ec=INK2, size=8.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2",
                                    fc=fc, ec=ec, lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                                     color=INK2, lw=1.1))
    ax.text(0, 40, "Stage 1: scan (one seed, not a final seed)", fontsize=10, color=BLUE,
            fontweight="bold")
    ax.text(55, 40, "Stage 2: main experiment", fontsize=10, color=ORANGE, fontweight="bold")
    box(0, 22, 24, 13, "47 cells x 3 rates\n3e-3, 1e-2, 3e-2\n4,000 steps each\n141 runs, about 8 h")
    box(29, 22, 24, 13, "each cell takes the rate\nwith the lowest\nvalidation bpc\n(diverged rates excluded)")
    box(29, 3, 24, 13, "only if the best rate is at\nan end of the grid: one more\nrun, a factor of 3 beyond\n(1e-3 or 9e-2)")
    box(55, 22, 24, 13, "47 cells x 5 final seeds\nat the chosen rate\n235 runs, about 14 h")
    box(82, 22, 24, 13, "analysis (section 9)\nand paper. Rates and\nthe scan table are\nreported", fc="#fbeee8", ec=ORANGE)
    arrow(24.4, 28.5, 28.6, 28.5)
    arrow(53.4, 28.5, 54.6, 28.5)
    arrow(41, 21.6, 41, 16.4)
    arrow(53.4, 13, 62, 21.6)
    arrow(79.4, 28.5, 81.6, 28.5)
    save(fig, "fig5_protocol.png")


if __name__ == "__main__":
    fig_grid()
    fig_geometry()
    fig_mask()
    fig_floor()
    fig_protocol()
    print("figures written to", OUT)
