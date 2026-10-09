"""Independent recomputation of the approved tests (claims A-D) from results_long.csv."""
import os, numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(HERE + '/../..')
T = REPO + '/results/analysis/tables'
L = pd.read_csv(T + '/results_long.csv'); S2 = pd.read_csv(T + '/test2_gaps.csv'); S3 = pd.read_csv(T + '/test3_trend.csv')
LP = ['lp_1', 'lp_1.5', 'lp_2', 'lp_3', 'lp_4', 'lp_8', 'lp_inf']; Ds = [2, 4, 8, 12, 16]
w = L[L.arm != 'uniform'].pivot_table(index=['seed', 'D'], columns='arm', values='val_bpc')
def tt(x):
    x = np.asarray(x, float); n = len(x); m = x.mean(); se = x.std(ddof=1) / np.sqrt(n); tcrit = stats.t.ppf(.975, n - 1)
    t = m / se; return m, m - tcrit * se, m + tcrit * se, t, 2 * stats.t.sf(abs(t), n - 1)
def holm(p):
    p = np.asarray(p); o = np.argsort(p); out = np.empty_like(p); run = 0
    for rank, i in enumerate(o):
        run = max(run, (len(p) - rank) * p[i]); out[i] = min(1, run)
    return out
mx = {'mean_ci': 0, 't': 0, 'p': 0}
def upd(**k):
    for a, b in k.items(): mx[a] = max(mx[a], b)
for D in Ds:
    rows = []
    for a in LP:
        g = (w[a] - w['dot_tied']).xs(D, level='D').values; rows.append(tt(g))
    ph = holm([r[4] for r in rows])
    for a, r, h in zip(LP, rows, ph):
        s = S2[(S2.D == D) & (S2.comparison == f'{a} - dot_tied')].iloc[0]
        upd(mean_ci=max(abs(s['mean'] - r[0]), abs(s.ci_low - r[1]), abs(s.ci_high - r[2])), t=abs(s.t - r[3]), p=max(abs(s.p - r[4]), abs(s.p_holm - h)))
for D in Ds:
    g = (w['dot_untied'] - w['dot_tied']).xs(D, level='D').values; r = tt(g)
    s = S2[(S2.D == D) & (S2.comparison == 'dot_untied - dot_tied')].iloc[0]; upd(mean_ci=abs(s['mean'] - r[0]), p=abs(s.p - r[4]))
u = L[L.arm == 'dot_untied_unmasked'].sort_values('seed').val_bpc.values - L[(L.arm == 'dot_untied') & (L.D == 16)].sort_values('seed').val_bpc.values
r = tt(u); s = S2[S2.comparison == 'dot_untied_unmasked - dot_untied'].iloc[0]; upd(mean_ci=abs(s['mean'] - r[0]), p=abs(s.p - r[4]))
sl = {}
for a in LP:
    sl[a] = [np.polyfit(np.log2(Ds), [(w[a] - w['dot_tied']).loc[(sd, D)] for D in Ds], 1)[0] for sd in range(1, 6)]
sl['mean'] = list(np.mean([sl[a] for a in LP], axis=0))
for k, v in sl.items():
    r = tt(v); name = 'mean of the seven arms' if k == 'mean' else k; s = S3[S3.arm == name].iloc[0]
    upd(mean_ci=max(abs(s['mean'] - r[0]), abs(s.ci_low - r[1])), p=abs(s.p - r[4]))
print('max abs difference, stored tables vs independent recomputation (the CSV files carry about six significant digits):', {k: float(f'{v:.2g}') for k, v in mx.items()})
sig = S2[S2.family.str.startswith('Holm')]; sig = sig.assign(sig=sig.p_holm < .05)
print('Holm-significant cells:', int(sig.sig.sum()), 'of', len(sig), ' by D:', sig.groupby('D').sig.sum().to_dict())
print('positive mean gaps:', int((sig['mean'] > 0).sum()), 'of', len(sig), '; negative Holm-significant:', int(((sig['mean'] < 0) & sig.sig).sum()))
print('uncorrected p<0.05:', int((sig.p < .05).sum()))
print('test3 per-arm negative means:', int((S3['mean'] < 0).sum()), 'of', len(S3), '; p<.05:', S3[S3.p < .05].arm.tolist())
