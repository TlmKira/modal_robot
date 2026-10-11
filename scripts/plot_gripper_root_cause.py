"""Show the omitted supervision and real predicted gripper commands."""
import json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];full=json.loads((root/'evaluation/offline_recovery_summary.json').read_text())['metrics'];late=json.loads((root/'evaluation/offline_late_phase_summary.json').read_text());groups=[]
for split in ['train','validation']:
 groups.append((split+' early',next(r for r in full if r['split']==split and r['horizon']==1)))
 for phase in ['grasp','lift']:groups.append((split+' '+phase,next(r for r in late if r['split']==split and r['phase_anchor']==phase and r['horizon']==1)))
fig,axes=plt.subplots(1,2,figsize=(13,4));x=np.arange(len(groups))
for side,offset in [('left',-.18),('right',.18)]:axes[0].bar(x+offset,[r[f'{side}_gripper_mae'] for _,r in groups],width=.36,label=side)
axes[0].set(xticks=x,xticklabels=[n for n,_ in groups],ylabel='First-step gripper MAE (normalized)');axes[0].tick_params(axis='x',rotation=30);axes[0].legend();axes[0].grid(axis='y')
expert=np.stack(pd.read_parquet(root/'datasets/lerobot/lift_pot_train_v21/data/chunk-000/episode_000020.parquet')['action'].to_list())
axes[1].plot(np.arange(len(expert))*.06,expert[:,6],color='k',label='Expert target (diagnostic only)')
for tag in ['recovery_h5_train','recovery_h40_train','recovery_h5_true_d16_train']:
 q=np.load(root/f'evaluation/lift_pot_gr00t_{tag}_seed1_actions.npz')['executed'];axes[1].plot(np.arange(len(q))*.06,q[:,6],label=tag)
axes[1].set(xlabel='Simulated time (s), nominal 60ms source cadence',ylabel='Left gripper command, 0=closed / 1=open');axes[1].legend(fontsize=7);axes[1].grid();fig.tight_layout();fig.savefig(root/'results/gripper_coverage_failure.png',dpi=160)
