"""Descriptive plots of wall-clock training time. Reads results/ only."""
import os, glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(HERE, '..', '..')) + '/results'
rows = []
for d in glob.glob(R + '/main/*/'):
    r = json.load(open(d + 'result.json'))
    rows.append(dict(arm=r['arm'], D=r['D'], seed=r['seed'], wall=r['wall_time_s'] / 60, bpc=r['final_val_bpc']))
t = pd.DataFrame(rows)
Ds = [2, 4, 8, 12, 16]
LP = ['lp_1', 'lp_1.5', 'lp_2', 'lp_3', 'lp_4', 'lp_8', 'lp_inf']
PLAB = {'lp_1': 'p = 1', 'lp_1.5': 'p = 1.5', 'lp_2': 'p = 2', 'lp_3': 'p = 3', 'lp_4': 'p = 4', 'lp_8': 'p = 8', 'lp_inf': 'p = ∞'}
BLUES = ['#0b2f66', '#1b4f9c', '#2a78d6', '#5a9be0', '#86b6e8', '#aecdf0', '#cfe0f6']
COL = dict(zip(LP, BLUES)); ORANGE, VIOLET, GREY, INK, MUTED = '#eb6834', '#4a3aa7', '#8a8a86', '#1c1c1b', '#6b6a66'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#bbbab5', 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e7e6e2', 'grid.linewidth': .7, 'figure.dpi': 150})
w = t.groupby(['arm', 'D']).wall.mean().unstack()
unif = t[t.arm == 'uniform'].wall.mean()
bpc = t.groupby(['arm', 'D']).bpc.mean().unstack()
x = np.array(Ds)

# Fig 6: minutes per run vs D
fig, ax = plt.subplots(figsize=(8, 5.2))
for a in LP: ax.plot(x, w.loc[a, Ds], '-o', color=COL[a], lw=2, ms=5, mec='white', mew=.8, label=PLAB[a])
ax.plot(x, w.loc['dot_tied', Ds], '-o', color=ORANGE, lw=3, ms=7, mec='white', mew=.8, label='tied dot product', zorder=5)
ax.plot(x, w.loc['dot_untied', Ds], '--s', color=VIOLET, lw=2, ms=5, mec='white', mew=.8, label='untied dot product', zorder=4)
ax.axhline(unif, color=GREY, lw=1.8, ls=':'); ax.text(2.05, unif + .8, f'uniform attention ({unif:.1f} min, no scores)', fontsize=8.5, color=MUTED)
ax.set_xscale('log', base=2); ax.set_xticks(Ds); ax.set_xticklabels(Ds); ax.minorticks_off(); ax.set_ylim(0, 52)
ax.set_xlabel('head size D (log scale)'); ax.set_ylabel('wall-clock minutes per run (mean of 5 seeds)')
ax.legend(ncol=2, fontsize=8.5, frameon=False, loc='upper left')
ax.set_title('Training time of one 4,000-step run', loc='left', fontsize=11.5)
fig.text(.01, .01, '4 runs in parallel, 1 thread each, 2020 Intel MacBook; chunked pure-PyTorch distance kernel.', fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(HERE + '/fig6_time_by_head_size.png'); plt.close(fig)

# Fig 7: ratio to tied dot at same D (cost of attention beyond the shared model: subtract uniform)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axs[0]
for a in LP: ax.plot(x, w.loc[a, Ds] / w.loc['dot_tied', Ds], '-o', color=COL[a], lw=2, ms=5, mec='white', mew=.8, label=PLAB[a])
ax.axhline(1, color=ORANGE, lw=2.5)
ax.set_xscale('log', base=2); ax.set_xticks(Ds); ax.set_xticklabels(Ds); ax.minorticks_off()
ax.set_xlabel('head size D (log scale)'); ax.set_ylabel('run time / tied-dot run time'); ax.set_title('Whole run', loc='left')
ax.legend(fontsize=8, frameon=False, ncol=2, loc='upper left')
ax = axs[1]
base = w.loc['dot_tied', Ds] - unif
for a in LP: ax.plot(x, (w.loc[a, Ds] - unif) / base, '-o', color=COL[a], lw=2, ms=5, mec='white', mew=.8)
ax.axhline(1, color=ORANGE, lw=2.5)
ax.set_xscale('log', base=2); ax.set_xticks(Ds); ax.set_xticklabels(Ds); ax.minorticks_off()
ax.set_xlabel('head size D (log scale)'); ax.set_ylabel('(run time − uniform run time) / same for tied dot')
ax.set_title('Attention-score part only (uniform run subtracted)', loc='left')
fig.suptitle('Cost relative to the tied dot product (orange line = 1)', x=.01, ha='left', fontsize=11.5)
fig.tight_layout(); fig.savefig(HERE + '/fig7_time_ratio.png'); plt.close(fig)

# Fig 8: bpc vs minutes
fig, ax = plt.subplots(figsize=(8, 5.2))
mk = {2: 'o', 4: 's', 8: '^', 12: 'D', 16: 'P'}
for a in LP + ['dot_tied', 'dot_untied']:
    c = COL.get(a, ORANGE if a == 'dot_tied' else VIOLET)
    for D in Ds:
        ax.scatter(w.loc[a, D], bpc.loc[a, D], color=c, marker=mk[D], s=46 if a.startswith('lp') else 70, edgecolor='white', lw=.8, zorder=4 if a.startswith('dot') else 3)
ax.set_xlabel('wall-clock minutes per run'); ax.set_ylabel('final validation bpc (mean of 5 seeds)')
from matplotlib.lines import Line2D
h = [Line2D([], [], marker=mk[D], ls='', color=GREY, label=f'D = {D}') for D in Ds] + [Line2D([], [], marker='o', ls='', color=ORANGE, label='tied dot'), Line2D([], [], marker='o', ls='', color=VIOLET, label='untied dot'), Line2D([], [], marker='o', ls='', color=BLUES[2], label='L^p (blues, as before)')]
ax.legend(handles=h, fontsize=8.5, frameon=False, ncol=2, loc='upper right')
ax.set_title('Quality against cost, one point per cell', loc='left', fontsize=11.5)
fig.tight_layout(); fig.savefig(HERE + '/fig8_bpc_vs_time.png'); plt.close(fig)

tab = w.loc[['dot_tied', 'dot_untied'] + LP, Ds].round(1); tab.to_csv(HERE + '/wall_minutes_by_cell.csv')
print(tab); print('ratio'); print((w.loc[LP, Ds] / w.loc['dot_tied', Ds]).round(2)); print('attention-only ratio'); print(((w.loc[LP, Ds] - unif) / (w.loc['dot_tied', Ds] - unif)).round(1))
print('unmasked', w.loc['dot_untied_unmasked', 16], 'uniform', unif)
