"""Descriptive plots of the bpc surface. Reads results/ only. No tests."""
import os, sys, json, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
R = REPO + '/results'
t = pd.read_csv(R + '/analysis/tables/results_long.csv')
t['D'] = t['D'].astype(float)
Ds = [2, 4, 8, 12, 16]
LP = ['lp_1', 'lp_1.5', 'lp_2', 'lp_3', 'lp_4', 'lp_8', 'lp_inf']
PLAB = {'lp_1': 'p = 1', 'lp_1.5': 'p = 1.5', 'lp_2': 'p = 2', 'lp_3': 'p = 3', 'lp_4': 'p = 4', 'lp_8': 'p = 8', 'lp_inf': 'p = ∞'}
INVP = {'lp_1': 1, 'lp_1.5': 2/3, 'lp_2': .5, 'lp_3': 1/3, 'lp_4': .25, 'lp_8': .125, 'lp_inf': 0}
BLUES = ['#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef']  # p=1 darkest .. inf lightest
BLUES = ['#0b2f66', '#1b4f9c', '#2a78d6', '#5a9be0', '#86b6e8', '#aecdf0', '#cfe0f6']
COL = dict(zip(LP, BLUES))
ORANGE, VIOLET, GREY, INK, MUTED = '#eb6834', '#4a3aa7', '#8a8a86', '#1c1c1b', '#6b6a66'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#bbbab5', 'axes.labelcolor': INK, 'text.color': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e7e6e2', 'grid.linewidth': .7, 'figure.dpi': 150})
out = HERE
mean = t.groupby(['arm', 'D']).val_bpc.mean().unstack()
sd = t.groupby(['arm', 'D']).val_bpc.std().unstack()
unif = t[t.arm == 'uniform'].val_bpc

# ---------- Fig 1: bpc vs D
fig, ax = plt.subplots(figsize=(8, 5.2))
x = np.array(Ds)
tied = mean.loc['dot_tied', Ds]
ax.fill_between(x, tied - sd.loc['dot_tied', Ds], tied + sd.loc['dot_tied', Ds], color=ORANGE, alpha=.13, lw=0)
for a in LP:
    ax.plot(x, mean.loc[a, Ds], '-o', color=COL[a], lw=2, ms=5, label=PLAB[a], mec='white', mew=.8)
ax.plot(x, tied, '-o', color=ORANGE, lw=3, ms=7, label='tied dot product', mec='white', mew=.8, zorder=5)
ax.plot(x, mean.loc['dot_untied', Ds], '--s', color=VIOLET, lw=2, ms=5, label='untied dot product', mec='white', mew=.8, zorder=4)
ax.plot([16], [mean.loc['dot_untied_unmasked', 16]], 'D', color=VIOLET, ms=7, mfc='white', mew=1.8, label='untied, unmasked (D = 16)', zorder=6)
ax.set_xscale('log', base=2); ax.set_xticks(Ds); ax.set_xticklabels(Ds); ax.minorticks_off()
ax.set_xlabel('head size D (log scale)'); ax.set_ylabel('final validation bpc (mean of 5 seeds)')
fig.text(.01, .01, f'Uniform-attention control: {unif.mean():.3f} (off the chart). Shaded band: ±1 seed sd of the tied dot product.', fontsize=8.5, color=MUTED)
ax.legend(ncol=2, fontsize=8.5, frameon=False, loc='upper right')
ax.set_title('Validation bpc by head size (lower is better)', loc='left', fontsize=11.5)
fig.tight_layout(rect=(0,.04,1,1)); fig.savefig(out + '/fig1_bpc_by_head_size.png'); plt.close(fig)

# ---------- Fig 2: heatmap of paired gap to tied dot
w = t[t.arm != 'uniform'].pivot_table(index=['seed', 'D'], columns='arm', values='val_bpc')
gap = w.sub(w['dot_tied'], axis=0)
gm = gap.groupby('D').mean().T
rows = LP + ['dot_untied']
M = gm.loc[rows, Ds].values
fig, ax = plt.subplots(figsize=(7.2, 4.6))
norm = TwoSlopeNorm(vmin=-0.02, vcenter=0, vmax=0.05)
im = ax.imshow(M, cmap='RdBu_r', norm=norm, aspect='auto')
ax.set_xticks(range(5)); ax.set_xticklabels([f'D = {d}' for d in Ds]); ax.set_yticks(range(len(rows)))
ax.set_yticklabels([PLAB[a] for a in LP] + ['untied dot'])
ax.grid(False)
for i in range(M.shape[0]):
    for j in range(5):
        v = M[i, j]; ax.text(j, i, f'{v:+.3f}', ha='center', va='center', fontsize=9, color='white' if abs(v) > .035 else INK)
ax.axhline(len(LP) - .5, color='white', lw=3)
cb = fig.colorbar(im, ax=ax, pad=.02); cb.set_label('mean paired gap to the tied dot product (bpc)\nred = worse than tied dot, blue = better')
ax.set_title('Gap to the tied dot product (same seed, same D)', loc='left', fontsize=11.5)
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig(out + '/fig2_gap_heatmap.png'); plt.close(fig)

# ---------- Fig 3: bpc vs 1/p, one panel per D, all seeds shown
fig, axs = plt.subplots(1, 5, figsize=(14, 3.9), sharey=True)
for ax, D in zip(axs, Ds):
    sub = t[t.D == D]
    for a in LP:
        v = sub[sub.arm == a].val_bpc.values
        ax.scatter([INVP[a]] * len(v), v, s=14, color=COL[a], alpha=.8, lw=0)
    ax.plot([INVP[a] for a in LP], [mean.loc[a, D] for a in LP], '-', color='#2a78d6', lw=1.6)
    ax.scatter([INVP[a] for a in LP], [mean.loc[a, D] for a in LP], s=36, color=[COL[a] for a in LP], edgecolor='white', zorder=4)
    ax.axhline(mean.loc['dot_tied', D], color=ORANGE, lw=2.2)
    ax.axhline(mean.loc['dot_untied', D], color=VIOLET, lw=1.6, ls='--')
    ax.set_title(f'D = {D}', loc='left', fontsize=10.5)
    ax.set_xlabel('1/p  (right = p = 1, left = p = ∞)')
    ax.set_xticks([0, .25, .5, 1]); ax.set_xticklabels(['0', '¼', '½', '1'])
axs[0].set_ylabel('final validation bpc')
fig.text(.01, .01, 'Orange line: tied dot product (mean). Violet dashed line: untied dot product (mean). Small dots: single seeds. Large dots: means of the L^p arms.', fontsize=8.5, color=MUTED)
fig.suptitle('Shape of the effect of p at each head size', x=.01, ha='left', fontsize=11.5)
fig.tight_layout(rect=(0,.05,1,1)); fig.savefig(out + '/fig3_bpc_by_invp.png'); plt.close(fig)

# ---------- Fig 4: gap-to-tied over training
def curves(arm, D):
    ds = []
    for s in range(1, 6):
        d = f'{R}/main/{arm}_D{D}_seed{s}/log.csv'
        lg = pd.read_csv(d).dropna(subset=['val_bpc']).set_index('step').val_bpc
        ds.append(lg)
    return pd.concat(ds, axis=1)
fig, axs = plt.subplots(1, 5, figsize=(15, 3.8), sharey=True)
for ax, D in zip(axs, Ds):
    base = curves('dot_tied', D)
    for a in LP:
        g = (curves(a, D) - base.values).mean(axis=1)
        ax.plot(g.index, g.values, '-o', color=COL[a], lw=1.6, ms=3, label=PLAB[a])
    g = (curves('dot_untied', D) - base.values).mean(axis=1)
    ax.plot(g.index, g.values, '--s', color=VIOLET, lw=1.5, ms=3, label='untied dot')
    ax.axhline(0, color=ORANGE, lw=2.2)
    ax.set_title(f'D = {D}', loc='left'); ax.set_xlabel('training step'); ax.set_ylim(-.06, .07)
    ax.set_xticks([500, 2000, 4000])
axs[0].set_ylabel('mean paired gap to tied dot (bpc)')
axs[0].legend(fontsize=7.5, frameon=False, ncol=2, loc='lower right')
fig.text(.01, .01, 'Orange line: tied dot product (gap = 0 by definition). Above the line = worse than tied dot. Mean over 5 seeds at each evaluation step.', fontsize=8.5, color=MUTED)
fig.suptitle('When does the gap to the tied dot product appear?', x=.01, ha='left', fontsize=11.5)
fig.tight_layout(rect=(0,.05,1,1)); fig.savefig(out + '/fig4_gap_over_training.png'); plt.close(fig)

# ---------- Fig 5: scale: uniform vs everything
fig, ax = plt.subplots(figsize=(7.5, 2.6))
vals = [('uniform attention', unif.mean(), GREY), ('best L^p cell', min(mean.loc[a, D] for a in LP for D in Ds), '#2a78d6'),
        ('worst L^p cell', max(mean.loc[a, D] for a in LP for D in Ds), '#2a78d6'),
        ('tied dot, mean over D', mean.loc['dot_tied', Ds].mean(), ORANGE), ('untied dot, mean over D', mean.loc['dot_untied', Ds].mean(), VIOLET)]
for i, (n, v, c) in enumerate(vals):
    ax.barh(i, v - 2.0, left=2.0, color=c, height=.6); ax.text(v + .01, i, f'{v:.3f}', va='center', fontsize=9)
ax.set_yticks(range(len(vals))); ax.set_yticklabels([v[0] for v in vals]); ax.invert_yaxis(); ax.set_xlim(2.0, 3.15)
ax.set_xlabel('validation bpc (axis starts at 2.0)'); ax.grid(axis='y', visible=False)
ax.set_title('Scale: how much can attention move the result?', loc='left', fontsize=11)
fig.tight_layout(); fig.savefig(out + '/fig5_scale_vs_uniform.png'); plt.close(fig)
print('done')
