"""Collect the small data files that the paper's figures and tables are built from. Reads results/ only."""
import os, glob, json, shutil, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(HERE + '/../..'); R = REPO + '/results'; OUT = HERE + '/draft/data'
os.makedirs(OUT, exist_ok=True)
for f in ['results_long', 'cell_means', 'test2_gaps', 'test3_trend']:
    shutil.copy(f'{R}/analysis/tables/{f}.csv', f'{OUT}/{f}.csv')
shutil.copy(f'{R}/scan_table.csv', f'{OUT}/scan_table.csv'); shutil.copy(f'{R}/calibration/calibration_summary.csv', f'{OUT}/calibration_summary.csv')
cur, wall = [], []
for d in sorted(glob.glob(R + '/main/*/')):
    r = json.load(open(d + 'result.json')); lg = pd.read_csv(d + 'log.csv').dropna(subset=['val_bpc'])
    for s, v in zip(lg.step, lg.val_bpc): cur.append((r['arm'], r['D'], r['seed'], int(s), v))
    wall.append((r['arm'], r['D'], r['seed'], r['wall_time_s'] / 60, r['final_train_loss']))
pd.DataFrame(cur, columns=['arm', 'D', 'seed', 'step', 'val_bpc']).to_csv(f'{OUT}/curves.csv', index=False)
pd.DataFrame(wall, columns=['arm', 'D', 'seed', 'wall_min', 'final_train_loss_nats']).to_csv(f'{OUT}/wall.csv', index=False)
print(os.listdir(OUT))
