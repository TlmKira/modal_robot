"""Compare real model against holding the current state; prevents misleading small MAE."""
import csv,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.spatial.transform import Rotation
from recovery_fk import poses
root=Path(__file__).resolve().parents[1];path=root/'results/offline_action_metrics.csv';rows=list(csv.DictReader(path.open()));cache={}
for r in rows:
 key=(r['split'],int(r['episode']));t=int(r['frame']);h=int(r['horizon'])
 if key not in cache:cache[key]=pd.read_parquet(root/f'datasets/lerobot/lift_pot_{key[0]}_v21/data/chunk-000/episode_{key[1]:06d}.parquet')
 ep=cache[key];state=np.asarray(ep['observation.state'].iloc[t]);target=np.asarray(ep['action'].iloc[t+h-1]);err=np.abs(state-target);ps,pt=poses(state),poses(target)
 for side,i,off in [('left',0,0),('right',1,7)]:
  r[f'persistence_{side}_joint_mae_rad']=float(err[off:off+6].mean());r[f'persistence_{side}_gripper_mae']=float(err[off+6]);r[f'persistence_{side}_tcp_error_m']=float(np.linalg.norm(ps[i][:3,3]-pt[i][:3,3]));r[f'persistence_{side}_orientation_error_rad']=float(Rotation.from_matrix(ps[i][:3,:3].T@pt[i][:3,:3]).magnitude())
with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for split in ['train','validation']:
 for h in ['1','5','10','20','40']:
  rs=[r for r in rows if r['split']==split and r['horizon']==h];summary.append({'split':split,'horizon':int(h),**{k:float(np.mean([float(r[k]) for r in rs])) for k in rows[0] if 'error' in k or 'mae' in k}})
(root/'evaluation/persistence_comparison.json').write_text(json.dumps(summary,indent=2))
for r in summary: print(r['split'],r['horizon'],'model/baseline L/R joint',*[round(r[k],5) for k in ['left_joint_mae_rad','persistence_left_joint_mae_rad','right_joint_mae_rad','persistence_right_joint_mae_rad']])
