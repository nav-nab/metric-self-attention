"""Build the figures, tables and number macros of the paper from the exported data files.
Run from the paper folder:  python build_assets.py"""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__)); D_ = HERE + '/data'; FIG = HERE + '/figures'; TAB = HERE + '/tables'
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
L = pd.read_csv(D_ + '/results_long.csv'); CM = pd.read_csv(D_ + '/cell_means.csv'); T2 = pd.read_csv(D_ + '/test2_gaps.csv')
T3 = pd.read_csv(D_ + '/test3_trend.csv'); SC = pd.read_csv(D_ + '/scan_table.csv'); CAL = pd.read_csv(D_ + '/calibration_summary.csv')
CUR = pd.read_csv(D_ + '/curves.csv'); WALL = pd.read_csv(D_ + '/wall.csv')
Ds = [2, 4, 8, 12, 16]
LP = ['lp_1', 'lp_1.5', 'lp_2', 'lp_3', 'lp_4', 'lp_8', 'lp_inf']
PLAB = {'lp_1': '$p=1$', 'lp_1.5': '$p=1.5$', 'lp_2': '$p=2$', 'lp_3': '$p=3$', 'lp_4': '$p=4$', 'lp_8': '$p=8$', 'lp_inf': r'$p=\infty$'}
PLAB_PLAIN = {'lp_1': 'p = 1', 'lp_1.5': 'p = 1.5', 'lp_2': 'p = 2', 'lp_3': 'p = 3', 'lp_4': 'p = 4', 'lp_8': 'p = 8', 'lp_inf': 'p = ∞'}
INVP = {'lp_1': 1, 'lp_1.5': 2 / 3, 'lp_2': .5, 'lp_3': 1 / 3, 'lp_4': .25, 'lp_8': .125, 'lp_inf': 0}
BLUES = ['#0b2f66', '#1b4f9c', '#2a78d6', '#5a9be0', '#86b6e8', '#aecdf0', '#cfe0f6']
COL = dict(zip(LP, BLUES)); ORANGE, VIOLET, GREY, INK, MUTED = '#eb6834', '#4a3aa7', '#8a8a86', '#1c1c1b', '#6b6a66'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': '#999', 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e7e6e2', 'grid.linewidth': .6,
                     'pdf.fonttype': 42, 'axes.titlesize': 8.5, 'legend.fontsize': 7, 'figure.dpi': 150})
N = {}   # number macros
def fmt(x, d=3): return f'{x:.{d}f}'
mean = L.groupby(['arm', 'D']).val_bpc.mean().unstack(); sd = L.groupby(['arm', 'D']).val_bpc.std().unstack()
unif = L[L.arm == 'uniform'].val_bpc
w = L[L.arm != 'uniform'].pivot_table(index=['seed', 'D'], columns='arm', values='val_bpc')

# ---------------- numbers
cross = [a for a in LP + ['dot_tied', 'dot_untied']]
allcross = np.array([mean.loc[a, D] for a in cross for D in Ds])
N['bpcMin'] = fmt(allcross.min()); N['bpcMax'] = fmt(allcross.max()); N['bpcRange'] = fmt(allcross.max() - allcross.min(), 3)
N['uniformBpc'] = fmt(unif.mean()); N['attnEffect'] = fmt(unif.mean() - allcross.max(), 2)
lpcells = np.array([mean.loc[a, D] for a in LP for D in Ds]); N['lpMin'] = fmt(lpcells.min()); N['lpMax'] = fmt(lpcells.max())
N['rangeFraction'] = f'{100 * (allcross.max() - allcross.min()) / (unif.mean() - allcross.min()):.0f}'
N['tiedDtwo'] = fmt(mean.loc['dot_tied', 2]); N['tiedDsixteen'] = fmt(mean.loc['dot_tied', 16])
N['untiedDsixteen'] = fmt(mean.loc['dot_untied', 16]); N['unmaskedDsixteen'] = fmt(mean.loc['dot_untied_unmasked', 16])
N['medianSeedSd'] = fmt(CM.sd_bpc.median(), 4); N['minSeedSd'] = fmt(CM.sd_bpc.min(), 4); N['maxSeedSd'] = fmt(CM.sd_bpc.max(), 4)
for D, nm in zip(Ds, ['Two', 'Four', 'Eight', 'Twelve', 'Sixteen']):
    N['spreadP' + nm] = fmt(max(mean.loc[a, D] for a in LP) - min(mean.loc[a, D] for a in LP), 3)
S2 = T2[T2.family.str.startswith('Holm')].copy(); S2['sig'] = S2.p_holm < .05
N['nPositive'] = str(int((S2['mean'] > 0).sum())); N['nHolm'] = str(int(S2.sig.sum()))
bd = S2.groupby('D').sig.sum().to_dict(); N['nHolmTwo'] = str(bd[2]); N['nHolmOther'] = str(int(S2[S2.D != 2].sig.sum()))
N['nHolmFour'] = str(bd[4]); N['nHolmEight'] = str(bd[8]); N['nHolmTwelve'] = str(bd[12]); N['nHolmSixteen'] = str(bd[16])
N['nUncorr'] = str(int((S2.p < .05).sum())); N['nNegative'] = str(int((S2['mean'] < 0).sum()))
N['gapTwoMin'] = fmt(S2[S2.D == 2]['mean'].min()); N['gapTwoMax'] = fmt(S2[S2.D == 2]['mean'].max())
oth = S2[S2.D != 2]; N['gapOtherMedian'] = fmt(oth['mean'].median(), 3); N['gapOtherMin'] = fmt(oth['mean'].min(), 3); N['gapOtherMax'] = fmt(oth['mean'].max(), 3)
ex = S2[S2['mean'] < 0].iloc[0]; N['exceptionGap'] = fmt(ex['mean'], 3); N['exceptionP'] = fmt(ex.p, 2); N['exceptionCIlo'] = fmt(ex.ci_low, 3); N['exceptionCIhi'] = fmt(ex.ci_high, 3)
UT = T2[T2.comparison == 'dot_untied - dot_tied'].set_index('D')
N['untiedMin'] = fmt(UT['mean'].min()); N['untiedMax'] = fmt(UT['mean'].max()); N['untiedEightP'] = fmt(UT.loc[8, 'p'], 3); N['untiedEight'] = fmt(UT.loc[8, 'mean'])
N['untiedSixteen'] = fmt(UT.loc[16, 'mean']); N['untiedSixteenP'] = fmt(UT.loc[16, 'p'], 3)
MK = T2[T2.comparison == 'dot_untied_unmasked - dot_untied'].iloc[0]
N['maskDiff'] = fmt(MK['mean'], 3); N['maskLo'] = fmt(MK.ci_low, 3); N['maskHi'] = fmt(MK.ci_high, 3); N['maskP'] = fmt(MK.p, 2)
M3 = T3[T3.arm == 'mean of the seven arms'].iloc[0]
N['slopeMean'] = fmt(M3['mean'], 4); N['slopeLo'] = fmt(M3.ci_low, 4); N['slopeHi'] = fmt(M3.ci_high, 4); N['slopeT'] = fmt(M3.t, 2); N['slopeP'] = fmt(M3.p, 4)
A3 = T3[T3.arm.isin(LP)]; N['slopeArmsSig'] = str(int((A3.p < .05).sum())); N['slopeArmsNeg'] = str(int((A3['mean'] < 0).sum()))
N['slopeMinArm'] = fmt(A3['mean'].max(), 4); N['slopeMaxArm'] = fmt(A3['mean'].min(), 4)
nonsig = A3[A3.p >= .05]; N['slopeNonsigList'] = ' and '.join(PLAB[a] for a in nonsig.arm)
N['slopeNonsigP'] = ' and '.join(fmt(p, 2) for p in nonsig.p)
# run time
wm = WALL.groupby(['arm', 'D']).wall_min.mean().unstack(); uni_t = WALL[WALL.arm == 'uniform'].wall_min.mean()
N['uniformMin'] = fmt(uni_t, 1)
dots = np.array([wm.loc[a, D] for a in ['dot_tied', 'dot_untied'] for D in Ds]); N['dotMinLo'] = fmt(dots.min(), 1); N['dotMinHi'] = fmt(dots.max(), 1)
lpt = np.array([wm.loc[a, D] for a in LP for D in Ds]); N['lpMinLo'] = fmt(lpt.min(), 1); N['lpMinHi'] = fmt(lpt.max(), 1)
rat = (wm.loc[LP, Ds] / wm.loc['dot_tied', Ds]); N['ratioTwoLo'] = fmt(rat[2].min(), 1); N['ratioTwoHi'] = fmt(rat[2].max(), 1)
N['ratioSixteenLo'] = fmt(rat[16].min(), 1); N['ratioSixteenHi'] = fmt(rat[16].max(), 1)
N['ratioPoneSixteen'] = fmt(rat.loc['lp_1', 16], 1); N['ratioPeightSixteen'] = fmt(rat.loc['lp_8', 16], 1)
N['totalHoursMain'] = fmt(WALL.wall_min.sum() / 60, 0)
# early training
C = CUR.copy(); C['D'] = C.D.astype(float)
base = C[C.arm == 'dot_tied'].set_index(['D', 'seed', 'step']).val_bpc
G = {}
for a in LP:
    x = C[C.arm == a].set_index(['D', 'seed', 'step']).val_bpc - base
    G[a] = x.groupby(['D', 'step']).mean().unstack('step')
steps = sorted(C.step.unique())
ahead = {s: int(sum((G[a][s] < 0).sum() for a in LP)) for s in steps}
meangap = {s: float(np.mean([G[a][s].values for a in LP])) for s in steps}
N['aheadFive'] = str(ahead[500]); N['aheadThousand'] = str(ahead[1000]); N['aheadFinal'] = str(ahead[4000])
N['meanGapFive'] = fmt(-meangap[500], 3); N['meanGapThousand'] = fmt(meangap[1000], 3); N['meanGapTwo'] = fmt(meangap[2000], 3); N['meanGapFinal'] = fmt(meangap[4000], 3)
g5 = np.array([G[a][500].values for a in LP]); N['fiveMin'] = fmt(-g5.max(), 3); N['fiveMax'] = fmt(-g5.min(), 3)
# learning rate
chosen = SC[SC.chosen]; N['nChosenThree'] = str(int((chosen.lr == 0.003).sum())); N['nCells'] = str(len(chosen))
piv = SC.pivot_table(index='cell', columns='lr', values='val_bpc')
N['marginOverThousandMedian'] = fmt((piv[0.001] - piv[0.003]).median(), 3); N['marginOverTenMedian'] = fmt((piv[0.01] - piv[0.003]).median(), 3)
N['nCalibLayers'] = str(len(CAL))
N['attnEffectBest'] = fmt(unif.mean() - allcross.min(), 2)
for D, nm in zip(Ds, ['Two', 'Four', 'Eight', 'Twelve', 'Sixteen']): N['meanGap' + nm + 'D'] = fmt(S2[S2.D == D]['mean'].mean(), 3)
N['nTested'] = str(len(S2))
with open(HERE + '/numbers.tex', 'w') as f:
    for k, v in N.items(): f.write(f'\\newcommand{{\\{k}}}{{{v}}}\n')
json.dump(N, open(HERE + '/numbers.json', 'w'), indent=1)
print(json.dumps(N, indent=1)); print('ahead by step', ahead); print('mean gap by step', {k: round(v, 4) for k, v in meangap.items()})

# ======================================================= figures
def style_ax(ax): ax.tick_params(length=2.5)
# ---- Fig 1 (main): bpc by D and gap heatmap
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={'width_ratios': [1.15, 1]})
ax = axs[0]; x = np.array(Ds); tied = mean.loc['dot_tied', Ds]
ax.fill_between(x, tied - sd.loc['dot_tied', Ds], tied + sd.loc['dot_tied', Ds], color=ORANGE, alpha=.14, lw=0)
for a in LP: ax.plot(x, mean.loc[a, Ds], '-o', color=COL[a], lw=1.3, ms=3, mec='white', mew=.5, label=PLAB_PLAIN[a])
ax.plot(x, tied, '-o', color=ORANGE, lw=2.2, ms=4.5, mec='white', mew=.6, label='tied dot', zorder=5)
ax.plot(x, mean.loc['dot_untied', Ds], '--s', color=VIOLET, lw=1.4, ms=3.2, mec='white', mew=.5, label='untied dot', zorder=4)
ax.plot([16], [mean.loc['dot_untied_unmasked', 16]], 'D', color=VIOLET, ms=4.5, mfc='white', mew=1.2, label='untied, unmasked', zorder=6)
ax.set_xscale('log', base=2); ax.set_xticks(Ds); ax.set_xticklabels(Ds); ax.minorticks_off(); style_ax(ax)
ax.set_xlabel('head size $D$'); ax.set_ylabel('final validation bpc (mean of 5 seeds)'); ax.set_title('(a) Validation bpc by head size', loc='left')
ax.legend(ncol=2, frameon=False, loc='upper right', handlelength=1.6, columnspacing=1.0)
ax = axs[1]; gap = w.sub(w['dot_tied'], axis=0); gm = gap.groupby('D').mean().T
rows = LP + ['dot_untied']; M = gm.loc[rows, Ds].values
im = ax.imshow(M, cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-0.02, vcenter=0, vmax=0.05), aspect='auto'); ax.grid(False)
ax.set_xticks(range(5)); ax.set_xticklabels(Ds); ax.set_yticks(range(len(rows))); ax.set_yticklabels([PLAB_PLAIN[a] for a in LP] + ['untied dot'])
sig = {(r.D, r.comparison.split(' - ')[0]): r.sig for r in S2.itertuples()}
for i, a in enumerate(rows):
    for j, D in enumerate(Ds):
        v = M[i, j]; star = '*' if sig.get((D, a), False) else ''
        ax.text(j, i, f'{v:+.3f}{star}', ha='center', va='center', fontsize=6.3, color='white' if abs(v) > .035 else INK)
ax.axhline(len(LP) - .5, color='white', lw=2); ax.set_xlabel('head size $D$'); ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title('(b) Gap to the tied dot (bpc)', loc='left')
fig.tight_layout(w_pad=1.2); fig.savefig(FIG + '/fig_surface.pdf'); plt.close(fig)

# ---- Fig 2 (main): gap over training, D = 2 and 16
def gap_curve(a, D): return G[a].loc[D]
def untied_gap(D):
    x = C[(C.arm == 'dot_untied')].set_index(['D', 'seed', 'step']).val_bpc - base
    return x.groupby(['D', 'step']).mean().unstack('step').loc[D]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
for ax, D in zip(axs, [2, 16]):
    for a in LP: g = gap_curve(a, D); ax.plot(g.index, g.values, '-o', color=COL[a], lw=1.2, ms=2.6, label=PLAB_PLAIN[a])
    g = untied_gap(D); ax.plot(g.index, g.values, '--s', color=VIOLET, lw=1.2, ms=2.6, label='untied dot')
    ax.axhline(0, color=ORANGE, lw=1.8); ax.set_title(f'$D={D}$', loc='left'); ax.set_xlabel('training step'); ax.set_xticks([500, 1000, 2000, 3000, 4000]); style_ax(ax)
axs[0].set_ylabel('paired gap to tied dot (bpc)'); axs[0].legend(frameon=False, ncol=2, loc='lower right', handlelength=1.6, fontsize=6.3)
fig.tight_layout(w_pad=1.0); fig.savefig(FIG + '/fig_training.pdf'); plt.close(fig)

# ---- Appendix: bpc vs 1/p
fig, axs = plt.subplots(1, 5, figsize=(7.2, 2.2), sharey=True)
for ax, D in zip(axs, Ds):
    sub = L[L.D == D]
    for a in LP: v = sub[sub.arm == a].val_bpc.values; ax.scatter([INVP[a]] * len(v), v, s=5, color=COL[a], alpha=.8, lw=0)
    ax.plot([INVP[a] for a in LP], [mean.loc[a, D] for a in LP], '-', color='#2a78d6', lw=0.9)
    ax.scatter([INVP[a] for a in LP], [mean.loc[a, D] for a in LP], s=16, color=[COL[a] for a in LP], edgecolor='white', lw=.4, zorder=4)
    ax.axhline(mean.loc['dot_tied', D], color=ORANGE, lw=1.4); ax.axhline(mean.loc['dot_untied', D], color=VIOLET, lw=1.0, ls='--')
    ax.set_title(f'$D={D}$', loc='left'); ax.set_xticks([0, .5, 1]); ax.set_xticklabels(['0', '½', '1']); ax.set_xlabel('$1/p$'); style_ax(ax)
axs[0].set_ylabel('final validation bpc'); fig.tight_layout(w_pad=.5); fig.savefig(FIG + '/fig_invp.pdf'); plt.close(fig)

# ---- Appendix: gap over training, all D
fig, axs = plt.subplots(1, 5, figsize=(7.2, 2.2), sharey=True)
for ax, D in zip(axs, Ds):
    for a in LP: g = gap_curve(a, D); ax.plot(g.index, g.values, '-', color=COL[a], lw=1.0)
    g = untied_gap(D); ax.plot(g.index, g.values, '--', color=VIOLET, lw=1.0); ax.axhline(0, color=ORANGE, lw=1.4)
    ax.set_title(f'$D={D}$', loc='left'); ax.set_xticks([500, 2000, 4000]); ax.set_xlabel('step'); style_ax(ax)
axs[0].set_ylabel('paired gap to tied dot (bpc)'); fig.tight_layout(w_pad=.5); fig.savefig(FIG + '/fig_training_all.pdf'); plt.close(fig)

# ---- Appendix: per-seed gaps against D, one panel per arm (test 3)
fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.6), sharey=True, sharex=True)
arms8 = LP + ['mean']
for ax, a in zip(axs.ravel(), arms8):
    for sd_ in range(1, 6):
        if a == 'mean': ys = [np.mean([(w[b] - w['dot_tied']).loc[(sd_, D)] for b in LP]) for D in Ds]
        else: ys = [(w[a] - w['dot_tied']).loc[(sd_, D)] for D in Ds]
        ax.plot(Ds, ys, '-o', color=GREY, lw=.6, ms=1.8, alpha=.8)
    ym = [np.mean([(w[b] - w['dot_tied']).loc[(s_, D)] for s_ in range(1, 6) for b in LP]) if a == 'mean' else np.mean([(w[a] - w['dot_tied']).loc[(s_, D)] for s_ in range(1, 6)]) for D in Ds]
    ax.plot(Ds, ym, '-o', color=COL.get(a, '#2a78d6'), lw=1.8, ms=3, mec='white', mew=.5); ax.axhline(0, color=ORANGE, lw=1.2)
    ax.set_xscale('log', base=2); ax.set_xticks(Ds); ax.set_xticklabels(Ds); ax.minorticks_off()
    ax.set_title('mean of seven arms' if a == 'mean' else PLAB_PLAIN[a], loc='left'); style_ax(ax)
for ax in axs[1]: ax.set_xlabel('$D$')
for ax in axs[:, 0]: ax.set_ylabel('gap to tied dot (bpc)')
fig.tight_layout(); fig.savefig(FIG + '/fig_gap_slopes.pdf'); plt.close(fig)

# ---- Appendix: run time
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
ax = axs[0]
for a in LP: ax.plot(Ds, wm.loc[a, Ds], '-o', color=COL[a], lw=1.2, ms=2.8, mec='white', mew=.4, label=PLAB_PLAIN[a])
ax.plot(Ds, wm.loc['dot_tied', Ds], '-o', color=ORANGE, lw=2, ms=3.8, label='tied dot'); ax.plot(Ds, wm.loc['dot_untied', Ds], '--s', color=VIOLET, lw=1.2, ms=2.8, label='untied dot')
ax.axhline(uni_t, color=GREY, ls=':', lw=1.3); ax.text(2.1, uni_t + 1, 'uniform attention', fontsize=6.5, color=MUTED)
ax.set_xscale('log', base=2); ax.set_xticks(Ds); ax.set_xticklabels(Ds); ax.minorticks_off(); ax.set_xlabel('head size $D$'); ax.set_ylabel('minutes per run'); ax.set_ylim(0, 52)
ax.set_title('(a) Wall-clock time of one run', loc='left'); ax.legend(frameon=False, ncol=2, loc='upper left', handlelength=1.5); style_ax(ax)
ax = axs[1]
for a in LP: ax.plot(Ds, rat.loc[a, Ds], '-o', color=COL[a], lw=1.2, ms=2.8, mec='white', mew=.4)
ax.axhline(1, color=ORANGE, lw=1.6); ax.set_xscale('log', base=2); ax.set_xticks(Ds); ax.set_xticklabels(Ds); ax.minorticks_off(); ax.set_xlabel('head size $D$')
ax.set_ylabel('ratio to the tied dot'); ax.set_title('(b) Run time relative to the tied dot', loc='left'); style_ax(ax)
fig.tight_layout(); fig.savefig(FIG + '/fig_time.pdf'); plt.close(fig)

# ======================================================= tables
ARMLAB = {'dot_tied': 'Tied dot', 'dot_untied': 'Untied dot', 'dot_untied_unmasked': 'Untied dot, unmasked', 'uniform': 'Uniform attention', **PLAB}
def write(name, s): open(f'{TAB}/{name}.tex', 'w').write(s)
# time table (main)
rows_ = ['dot_tied', 'dot_untied'] + LP
s = '\\begin{tabular}{lrrrrr}\n\\toprule\nArm & $D=2$ & $D=4$ & $D=8$ & $D=12$ & $D=16$ \\\\\n\\midrule\n'
for a in rows_: s += ARMLAB[a] + ' & ' + ' & '.join(f'{wm.loc[a, D]:.1f}' for D in Ds) + ' \\\\\n'
s += '\\midrule\nFastest $\\ell^p$ arm / tied dot & ' + ' & '.join(f'{rat[D].min():.1f}' for D in Ds) + ' \\\\\nSlowest $\\ell^p$ arm / tied dot & ' + ' & '.join(f'{rat[D].max():.1f}' for D in Ds) + ' \\\\\n'
s += '\\midrule\nUniform attention (no scores) & \\multicolumn{5}{c}{' + f'{uni_t:.1f}' + '} \\\\\n\\bottomrule\n\\end{tabular}\n'; write('tab_time', s)
# cell means
s = '\\begin{tabular}{lccccc}\n\\toprule\nArm & $D=2$ & $D=4$ & $D=8$ & $D=12$ & $D=16$ \\\\\n\\midrule\n'
for a in ['dot_tied', 'dot_untied'] + LP:
    s += ARMLAB[a] + ' & ' + ' & '.join(f'{mean.loc[a, D]:.3f} $\\pm$ {sd.loc[a, D]:.3f}' for D in Ds) + ' \\\\\n'
s += '\\midrule\n' + ARMLAB['dot_untied_unmasked'] + ' & \\multicolumn{4}{c}{} & ' + f'{mean.loc["dot_untied_unmasked", 16]:.3f} $\\pm$ {sd.loc["dot_untied_unmasked", 16]:.3f}' + ' \\\\\n'
s += 'Uniform attention & \\multicolumn{5}{c}{' + f'{unif.mean():.3f} $\\pm$ {unif.std():.3f}' + '} \\\\\n\\bottomrule\n\\end{tabular}\n'; write('tab_cellmeans', s)
# test 2 table
def f3(x): return f'{x:+.3f}'
def pf(p): return '$<0.001$' if p < 0.001 else f'{p:.3f}'
s = '\\begin{longtable}{rlrrrrrrrrr}\n\\toprule\n$D$ & Arm & Gap & 95\\% interval & $p$ & $p_{\\text{Holm}}$ & $s_1$ & $s_2$ & $s_3$ & $s_4$ & $s_5$ \\\\\n\\midrule\n\\endhead\n'
for r in S2.itertuples():
    a = r.comparison.split(' - ')[0]; st = '$^{*}$' if r.sig else ''
    s += f'{r.D} & {ARMLAB[a]} & {f3(r.mean)}{st} & [{r.ci_low:+.3f}, {r.ci_high:+.3f}] & {pf(r.p)} & {pf(r.p_holm)} & ' + ' & '.join(f3(getattr(r, f"seed_{k}")) for k in range(1, 6)) + ' \\\\\n'
    if r.Index != S2.index[-1] and S2.loc[r.Index + 1, 'D'] != r.D: s += '\\midrule\n'
s += '\\bottomrule\n\\end{longtable}\n'; write('tab_test2', s)
OT = T2[~T2.family.str.startswith('Holm')]
s = '\\begin{tabular}{rlrrrrrrrr}\n\\toprule\n$D$ & Comparison & Mean & 95\\% interval & $p$ & $s_1$ & $s_2$ & $s_3$ & $s_4$ & $s_5$ \\\\\n\\midrule\n'
for r in OT.itertuples():
    nm = 'Untied $-$ tied' if r.comparison.startswith('dot_untied - ') else 'Unmasked $-$ masked (untied)'
    s += f'{r.D} & {nm} & {f3(r.mean)} & [{r.ci_low:+.3f}, {r.ci_high:+.3f}] & {pf(r.p)} & ' + ' & '.join(f3(getattr(r, f"seed_{k}")) for k in range(1, 6)) + ' \\\\\n'
s += '\\bottomrule\n\\end{tabular}\n'; write('tab_other', s)
# test 3
s = '\\begin{tabular}{lrrrrrrrrr}\n\\toprule\nArm & Slope & 95\\% interval & $t$ & $p$ & $s_1$ & $s_2$ & $s_3$ & $s_4$ & $s_5$ \\\\\n\\midrule\n'
for r in T3.itertuples():
    nm = 'Mean of the seven arms' if r.arm.startswith('mean') else ARMLAB[r.arm]
    if r.arm.startswith('mean'): s += '\\midrule\n'
    s += f'{nm} & {r.mean:+.4f} & [{r.ci_low:+.4f}, {r.ci_high:+.4f}] & {r.t:.2f} & {pf(r.p)} & ' + ' & '.join(f'{getattr(r, f"seed_{k}"):+.4f}' for k in range(1, 6)) + ' \\\\\n'
s += '\\bottomrule\n\\end{tabular}\n'; write('tab_test3', s)
# scan table
SCp = SC.pivot_table(index='cell', columns='lr', values='val_bpc'); chosen_lr = SC[SC.chosen].set_index('cell').lr
order = ['dot_tied', 'dot_untied', 'dot_untied_unmasked'] + LP
cells = []
for a in order:
    for D in (Ds if a != 'dot_untied_unmasked' else [16]): cells.append((a, D, f'{a}_D{D}'))
cells.append(('uniform', None, 'uniform'))
s = '\\begin{longtable}{llrrrrr}\n\\toprule\nArm & $D$ & $10^{-3}$ & $3\\cdot10^{-3}$ & $10^{-2}$ & $3\\cdot10^{-2}$ & Chosen \\\\\n\\midrule\n\\endhead\n'
for a, D, c in cells:
    row = SCp.loc[c]; vals = []
    for lr in [0.001, 0.003, 0.01, 0.03]:
        v = row.get(lr, np.nan)
        if np.isnan(v): vals.append('--')
        else: vals.append(('\\textbf{' + f'{v:.3f}' + '}') if abs(lr - chosen_lr[c]) < 1e-12 else f'{v:.3f}')
    s += f'{ARMLAB[a]} & {"--" if D is None else D} & ' + ' & '.join(vals) + f' & {chosen_lr[c]:g} \\\\\n'
s += '\\bottomrule\n\\end{longtable}\n'; write('tab_scan', s)
# calibration constants
s = '\\begin{tabular}{lrrrrr}\n\\toprule\nArm & $D=2$ & $D=4$ & $D=8$ & $D=12$ & $D=16$ \\\\\n\\midrule\n'
for a in order:
    cells_ = []
    for D in Ds:
        sub = CAL[(CAL.arm == a) & (CAL.D == D)].sort_values('layer')
        cells_.append('--' if len(sub) == 0 else ' / '.join(f'{v:.2f}' for v in sub.c))
    s += ARMLAB[a] + ' & ' + ' & '.join(cells_) + ' \\\\\n'
s += '\\bottomrule\n\\end{tabular}\n'; write('tab_calib', s)
dt = CAL[CAL.arm == 'dot_tied']
s = '\\begin{tabular}{lrrrrr}\n\\toprule\nLayer & $D=2$ & $D=4$ & $D=8$ & $D=12$ & $D=16$ \\\\\n\\midrule\n'
for ly in [0, 1]: s += f'{ly + 1} & ' + ' & '.join(f'{dt[(dt.D == D) & (dt.layer == ly)].c_zvar_sqrtD.iloc[0]:.2f}' for D in Ds) + ' \\\\\n'
s += '\\bottomrule\n\\end{tabular}\n'; write('tab_calibcheck', s)
print('assets built:', sorted(os.listdir(FIG)), sorted(os.listdir(TAB)))
# steps table (appendix)
s = '\\begin{tabular}{rrr}\n\\toprule\nStep & Cells with an $\\ell^p$ arm ahead (of 35) & Mean gap over the 35 cells (bpc) \\\\\n\\midrule\n'
for st in steps: s += f'{int(st)} & {ahead[st]} & {meangap[st]:+.3f} \\\\\n'
s += '\\bottomrule\n\\end{tabular}\n'; write('tab_steps', s)

# ---------------- revision: main-body summary tables
s = '\\begin{tabular}{rccccc}\n\\toprule\n$D$ & Tied dot & Untied $-$ tied & Mean gap & Range of gaps & Gap $>0$ / significant (of 7) \\\\\n\\midrule\n'
for D in Ds:
    sub = S2[S2.D == D]
    s += f"{D} & {mean.loc['dot_tied', D]:.3f} & {UT.loc[D, 'mean']:+.3f} & {sub['mean'].mean():+.3f} & ${sub['mean'].min():+.3f}$ to ${sub['mean'].max():+.3f}$ & {int((sub['mean'] > 0).sum())} / {int(sub.sig.sum())} \\\\\n"
s += '\\bottomrule\n\\end{tabular}\n'; write('tab_summary', s)
s = '\\begin{tabular}{lccc}\n\\toprule\nQuantity (bpc) & Estimate & 95\\% interval & $p$ \\\\\n\\midrule\n'
s += f"Slope of the gap per doubling of $D$ (mean of seven arms) & {M3['mean']:+.4f} & [{M3.ci_low:+.4f}, {M3.ci_high:+.4f}] & {M3.p:.4f} \\\\\n"
s += f"Unmasked minus masked untied dot, $D=16$ & {MK['mean']:+.3f} & [{MK.ci_low:+.3f}, {MK.ci_high:+.3f}] & {MK.p:.2f} \\\\\n"
s += '\\bottomrule\n\\end{tabular}\n'; write('tab_tests', s)
s = '\\begin{tabular}{lc}\n\\toprule\nQuantity & bpc \\\\\n\\midrule\n'
s += f"Lowest cell mean of the nine crossed arms & {allcross.min():.3f} \\\\\nHighest cell mean of the nine crossed arms & {allcross.max():.3f} \\\\\nRange & {allcross.max() - allcross.min():.3f} \\\\\n"
s += f"Uniform attention & {unif.mean():.3f} \\\\\nUniform attention minus highest crossed cell & {unif.mean() - allcross.max():.3f} \\\\\n"
s += f"Median seed standard deviation of a cell & {CM.sd_bpc.median():.4f} \\\\\n"
s += f"Tied dot, $D=2$ and $D=16$ & {mean.loc['dot_tied', 2]:.3f} and {mean.loc['dot_tied', 16]:.3f} \\\\\n"
s += f"Largest spread of the seven $\\ell^p$ arms within one $D$ ($D=16$) & {N['spreadPSixteen']} \\\\\nSmallest such spread ($D=4$) & {N['spreadPFour']} \\\\\n"
s += '\\bottomrule\n\\end{tabular}\n'; write('tab_scale', s)

# ---------------- revision 2: design grid and geometry figures
from matplotlib.patches import Rectangle
def fig_grid():
    rows = [('tied dot product (control)', 'tied', ORANGE)] + [(f'$\\ell^{{{p}}}$ distance', 'lp', BLUES[2]) for p in ['1', '1.5', '2', '3', '4', '8', '\\infty']] \
         + [('untied dot product (reference)', 'untied', ORANGE)]
    fig, ax = plt.subplots(figsize=(6.6, 3.9)); ax.set_xlim(-2.6, 7.4); ax.set_ylim(-3.1, 9.9); ax.axis('off'); ax.invert_yaxis(); ax.grid(False)
    for j, D in enumerate(Ds): ax.text(j + .5, -.35, f'$D={D}$', ha='center', va='bottom', fontsize=8)
    ax.plot([0, 5], [-.2, -.2], color='#999', lw=.7)
    for i, (lab, kind, col) in enumerate(rows):
        ax.text(-.1, i + .5, lab, ha='right', va='center', fontsize=7.5)
        for j in range(5): ax.add_patch(Rectangle((j + .06, i + .1), .88, .8, color=col, lw=0))
    ax.plot([5.2, 5.3, 5.3, 5.2], [1, 1, 8, 8], color='#777', lw=.8); ax.text(5.4, 4.5, 'crossed design:\n9 arms $\\times$ 5 head sizes\n= 45 cells', va='center', fontsize=7, color=MUTED)
    ax.add_patch(Rectangle((.06, 9.1), 4.88, .8, color='#b9b8b0', lw=0)); ax.text(2.5, 9.5, 'uniform control: one cell, the same for every $D$', ha='center', va='center', fontsize=7)
    ax.text(5.4, 9.5, 'extra: 1 cell', va='center', fontsize=7, color=MUTED)
    ax.add_patch(Rectangle((4.06, 10.1), .88, .8, color=VIOLET, lw=0)); ax.text(4.5, 10.5, '$D{=}16$', ha='center', va='center', fontsize=6.5, color='white')
    ax.text(-.1, 10.5, 'unmasked untied dot product', ha='right', va='center', fontsize=7.5); ax.text(5.4, 10.5, 'extra: 1 cell', va='center', fontsize=7, color=MUTED)
    ax.set_ylim(11.3, -1.3)
    fig.savefig(f'{FIG}/fig_grid.pdf', bbox_inches='tight'); plt.close(fig)
fig_grid()

def fig_levelsets():
    fig, axs = plt.subplots(1, 4, figsize=(6.8, 2.0)); t = np.linspace(0, 2 * np.pi, 400)
    xs = np.linspace(-1.2, 1.2, 400); X, Y = np.meshgrid(xs, xs)
    for ax in axs: ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax = axs[0]; q = np.array([.55, .35])
    for tau in np.linspace(-1.2, 1.2, 9): 
        n = q / np.linalg.norm(q); c = tau * n; d = np.array([-n[1], n[0]]); ax.plot(*np.array([c - 2 * d, c + 2 * d]).T, color=ORANGE, lw=.9)
    ax.annotate('', xy=q, xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=INK, lw=1)); ax.text(q[0] + .05, q[1] - .22, '$q$', fontsize=8)
    ax.set_title('dot product $q\\cdot k$\nlevel sets are lines', fontsize=7.5)
    for ax, p, lab in zip(axs[1:], [1, 2, np.inf], ['$\\ell^1$ distance', '$\\ell^2$ distance', '$\\ell^\\infty$ distance']):
        d = np.abs(X) + np.abs(Y) if p == 1 else (np.sqrt(X**2 + Y**2) if p == 2 else np.maximum(abs(X), abs(Y)))
        ax.contour(X, Y, d, levels=[.25, .5, .75, 1.0], colors=BLUES[2], linewidths=.9); ax.plot(0, 0, 'o', color=INK, ms=3); ax.text(.07, -.3, '$q$', fontsize=8, bbox=dict(fc='white', ec='none', pad=.5), zorder=6)
        ax.set_title(lab + '\nlevel sets are balls around $q$', fontsize=7.5)
    for ax in axs: 
        for s in ax.spines.values(): s.set_color('#bbb')
    fig.tight_layout(); fig.savefig(f'{FIG}/fig_levelsets.pdf', bbox_inches='tight'); plt.close(fig)
fig_levelsets()

def fig_interior():
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(3); P = rng.uniform(-1, 1, (60, 2)); P = P[np.hypot(*P.T) < 1][:13]
    c = P.mean(0); inner = int(np.argmin(np.hypot(*(P - c).T))); H = ConvexHull(P); assert inner not in H.vertices
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9))
    for ax in axs:
        ax.set_aspect('equal'); ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.3, 1.3); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values(): s.set_color('#bbb')
        hv = list(H.vertices) + [H.vertices[0]]; ax.plot(P[hv, 0], P[hv, 1], color=GREY, lw=.8, ls=':')
        ax.scatter(P[:, 0], P[:, 1], s=14, color=INK, zorder=3); ax.scatter(*P[inner], s=60, color=VIOLET, marker='*', zorder=4)
    ax = axs[0]
    for ang, col in zip([20, 135, 250], [ORANGE, '#c4501f', '#8e3a14']):
        q = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))]); w = int(np.argmax(P @ q)); n = q
        ax.annotate('', xy=q * 1.3, xytext=q * .7, arrowprops=dict(arrowstyle='->', color=col, lw=1.1))
        d = np.array([-n[1], n[0]]); cpt = P[w]; ax.plot(*np.array([cpt - 1.4 * d, cpt + 1.4 * d]).T, color=col, lw=.8, ls='--')
        ax.scatter(*P[w], s=34, facecolors='none', edgecolors=col, lw=1.2, zorder=5)
    ax.set_title('(a) dot product: the winner is always on the edge', fontsize=7.5)
    ax.text(P[inner][0] + .06, P[inner][1] - .2, 'never wins', fontsize=7, color=VIOLET)
    ax = axs[1]; q = None
    for rr in [.16, .12, .09]:
        for a_ in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            cand = P[inner] + rr * np.array([np.cos(a_), np.sin(a_)]); dd = np.hypot(*(P - cand).T); o_ = np.argsort(dd)
            if o_[0] == inner and dd[o_[1]] > 1.4 * dd[inner]: q = cand; break
        if q is not None: break
    assert q is not None; ax.plot(*q, 'x', color=BLUES[1], ms=7, mew=1.6, zorder=6)
    r = np.hypot(*(P[inner] - q)); order = np.argsort(np.hypot(*(P - q).T)); assert order[0] == inner
    r2 = np.hypot(*(P[order[1]] - q)); assert r2 > r * 1.05; rad = r; tt = np.linspace(0, 2 * np.pi, 200); ax.plot(q[0] + rad * np.cos(tt), q[1] + rad * np.sin(tt), color=BLUES[2], lw=1)
    ax.text(q[0] - .06, q[1] + .22, '$q$', fontsize=8, color=BLUES[1])
    ax.set_title('(b) distance: the nearest key wins', fontsize=7.5)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig_interior.pdf', bbox_inches='tight'); plt.close(fig)
fig_interior()

def fig_cloud():
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(3); P = rng.uniform(-1, 1, (60, 2)); P = P[np.hypot(*P.T) < 1][:13]
    c = P.mean(0); inner = int(np.argmin(np.hypot(*(P - c).T))); H = ConvexHull(P)
    fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.5))
    for ax in axs:
        ax.set_aspect('equal'); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.2, 1.2); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s_ in ax.spines.values(): s_.set_color('#bbb')
        hv = list(H.vertices) + [H.vertices[0]]; ax.plot(P[hv, 0], P[hv, 1], color=GREY, lw=.9, ls=':')
        ax.scatter(P[:, 0], P[:, 1], s=14, color=INK, zorder=3); ax.scatter(*P[inner], s=60, color=VIOLET, marker='*', zorder=4)
    ax = axs[0]
    for ang in [20, 135, 250]:
        q = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))]); w = int(np.argmax(P @ q))
        ax.annotate('', xy=q * 1.2, xytext=q * .75, arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.2))
        ax.scatter(*P[w], s=36, facecolors='none', edgecolors=ORANGE, lw=1.3, zorder=5)
    ax.set_title('(a) dot product: winners lie on the edge', fontsize=7.5)
    ax = axs[1]; q = None
    for rr in [.16, .12, .09]:
        for a_ in np.linspace(0, 2 * np.pi, 24, endpoint=False):
            cand = P[inner] + rr * np.array([np.cos(a_), np.sin(a_)]); dd = np.hypot(*(P - cand).T); o_ = np.argsort(dd)
            if o_[0] == inner and dd[o_[1]] > 1.4 * dd[inner]: q = cand; break
        if q is not None: break
    ax.plot(*q, 'x', color=BLUES[1], ms=7, mew=1.6, zorder=6); ax.scatter(*P[inner], s=60, color=VIOLET, marker='*', zorder=7)
    rad = np.hypot(*(P[inner] - q)); tt = np.linspace(0, 2 * np.pi, 200); ax.plot(q[0] + rad * np.cos(tt), q[1] + rad * np.sin(tt), color=BLUES[2], lw=1)
    ax.set_title('(b) distance: the nearest key wins', fontsize=7.5)
    fig.tight_layout(); fig.savefig(f'{FIG}/fig_cloud.pdf', bbox_inches='tight'); plt.close(fig)
fig_cloud()

# ---------------- revision 3: superposition appendix tables (margins on a circle, greedy packing)
rows = ['\\begin{tabular}{rccc}', '\\toprule', '$P$ & Dot product & Distance & Ratio \\\\', '\\midrule']
for P in [16, 32, 64, 128]:
    th = 2 * np.pi * np.arange(P) / P; K = np.c_[np.cos(th), np.sin(th)]; q = K[0]
    dot = K @ q; dist = -np.linalg.norm(K - q, axis=1)
    md = (dot[0] - np.sort(dot[1:])[-1]) / dot.std(); mt = (dist[0] - np.sort(dist[1:])[-1]) / dist.std()
    rows.append(f'{P} & {md:.4f} & {mt:.3f} & {mt / md:.1f} \\\\')
rows += ['\\bottomrule', '\\end{tabular}']; write('tab_margin', '\n'.join(rows) + '\n')
rng_p = np.random.default_rng(0)
def greedy_count(D_, eps, region, n=6000):
    pts = []
    for _ in range(n):
        x = rng_p.normal(size=D_); x /= np.linalg.norm(x)
        if region == 'ball': x *= rng_p.uniform() ** (1 / D_)
        if all(np.linalg.norm(x - p_) >= eps for p_ in pts): pts.append(x)
    return len(pts)
rows = ['\\begin{tabular}{rcccc}', '\\toprule', '$D$ & Separation & Keys on the sphere & Keys in the ball & Ratio \\\\', '\\midrule']
for D_ in [2, 3, 4]:
    for eps in [0.5, 0.35, 0.25]:
        a_ = greedy_count(D_, eps, 'sphere'); b_ = greedy_count(D_, eps, 'ball'); rows.append(f'{D_} & {eps} & {a_} & {b_} & {b_ / a_:.2f} \\\\')
    if D_ != 4: rows.append('\\addlinespace')
rows += ['\\bottomrule', '\\end{tabular}']; write('tab_pack', '\n'.join(rows) + '\n')
