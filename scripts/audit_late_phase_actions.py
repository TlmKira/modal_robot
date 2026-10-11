"""Supplement early full-chunk anchors with grasp/lift state observations near episode end."""
import json,csv,runpy,argparse
from pathlib import Path
import numpy as np,torch
from scipy.spatial.transform import Rotation
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.policy.gr00t_policy import Gr00tPolicy
from recovery_fk import poses
p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='checkpoints/lift_pot_2000/checkpoint-2000');p.add_argument('--dataset-version',default='v21');p.add_argument('--output-suffix',default='');a=p.parse_args()
root=Path(__file__).resolve().parents[1];cfg=runpy.run_path(str(root/'configs/aloha_gr00t_modality.py'))['config'];manifest=json.loads((root/'configs/dataset_split_manifest.json').read_text());policy=Gr00tPolicy(embodiment_tag='NEW_EMBODIMENT',model_path=str((root/a.checkpoint).resolve()),device='cuda:0');policy.model.action_head.num_inference_timesteps=4;torch.manual_seed(20261011);rows=[]
for split in ['train','validation']:
 loader=LeRobotEpisodeLoader(root/f'datasets/lerobot/lift_pot_{split}_{a.dataset_version}',cfg)
 for entry in manifest['splits'][split]:
  ep=loader[entry['episode_index']]
  real_length=len(ep)-(39 if 'terminal_pad' in a.dataset_version else 0)
  for phase,t in [('grasp',int(real_length*.75)),('lift',real_length-5)]:
   obs={'video':{k:np.asarray(ep[f'video.{k}'].iloc[t],dtype=np.uint8)[None,None] for k in cfg['video'].modality_keys},'state':{k:np.asarray(ep[f'state.{k}'].iloc[t],dtype=np.float32).reshape(1,1,-1) for k in cfg['state'].modality_keys},'language':{'annotation.human.task_description':[[str(ep['language.annotation.human.task_description'].iloc[t])]]}}
   acts,_=policy.get_action(obs);pred=np.concatenate([acts[k][0] for k in cfg['action'].modality_keys],axis=-1)
   for h in [1,5,10,20,40]:
    if t+h>real_length:continue
    target=np.concatenate([np.asarray(ep[f'action.{k}'].iloc[t+h-1]).reshape(-1) for k in cfg['action'].modality_keys]);err=np.abs(pred[h-1]-target);pp,pt=poses(pred[h-1]),poses(target);r={'split':split,'episode':entry['episode_index'],'frame':t,'phase_anchor':phase,'horizon':h}
    for i,side,offset in [(0,'left',0),(1,'right',7)]:r.update({f'{side}_joint_mae_rad':float(err[offset:offset+6].mean()),f'{side}_gripper_mae':float(err[offset+6]),f'{side}_tcp_error_m':float(np.linalg.norm(pp[i][:3,3]-pt[i][:3,3])),f'{side}_orientation_error_rad':float(Rotation.from_matrix(pp[i][:3,:3].T@pt[i][:3,:3]).magnitude())})
    rows.append(r)
  print(split,entry['episode_index'],flush=True)
with (root/f'results/offline_action_metrics_late_phase{a.output_suffix}.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
s=[]
for split in ['train','validation']:
 for phase in ['grasp','lift']:
  for h in [1,5,10,20,40]:
   rs=[r for r in rows if r['split']==split and r['phase_anchor']==phase and r['horizon']==h]
   if rs:s.append({'split':split,'phase_anchor':phase,'horizon':h,'samples':len(rs),**{k:float(np.mean([r[k] for r in rs])) for k in rows[0] if 'error' in k or 'mae' in k}})
(root/f'evaluation/offline_late_phase_summary{a.output_suffix}.json').write_text(json.dumps(s,indent=2))
