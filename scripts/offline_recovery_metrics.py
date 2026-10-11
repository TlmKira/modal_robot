"""Stratified real N1.7 prediction audit, preserving raw predictions and FK errors."""
import argparse, csv, json, runpy, time
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.policy.gr00t_policy import Gr00tPolicy
from recovery_fk import poses
p=argparse.ArgumentParser(); p.add_argument('--anchors',type=int,default=5); p.add_argument('--output-suffix',default=''); p.add_argument('--denoising-steps',type=int,default=4); p.add_argument('--checkpoint',default='checkpoints/lift_pot_2000/checkpoint-2000'); p.add_argument('--dataset-version',default='v21'); a=p.parse_args()
root=Path(__file__).resolve().parents[1]; config=runpy.run_path(str(root/'configs/aloha_gr00t_modality.py'))['config']
policy=Gr00tPolicy(embodiment_tag='NEW_EMBODIMENT',model_path=str((root/a.checkpoint).resolve()),device='cuda:0'); policy.model.action_head.num_inference_timesteps=a.denoising_steps
torch.manual_seed(20261010); rows=[]; predictions=[]; targets=[]; start=time.time()
manifest=json.loads((root/'configs/dataset_split_manifest.json').read_text())
for split,ds in [('train','train'),('validation','validation')]:
    loader=LeRobotEpisodeLoader(root/f'datasets/lerobot/lift_pot_{ds}_{a.dataset_version}',config)
    for entry in manifest['splits'][split]:
        ei=entry['episode_index']; ep=loader[ei]
        # Exclude horizons beyond episode end; sample full valid 40-step chunks.
        for t in np.linspace(0,len(ep)-40,a.anchors,dtype=int):
            obs={'video':{k:np.asarray(ep[f'video.{k}'].iloc[t],dtype=np.uint8)[None,None] for k in config['video'].modality_keys},'state':{k:np.asarray(ep[f'state.{k}'].iloc[t],dtype=np.float32).reshape(1,1,-1) for k in config['state'].modality_keys},'language':{'annotation.human.task_description':[[str(ep['language.annotation.human.task_description'].iloc[t])]]}}
            actions,_=policy.get_action(obs); pred=np.concatenate([actions[k][0] for k in config['action'].modality_keys],axis=-1)
            target=np.concatenate([np.stack(ep[f'action.{k}'].iloc[t:t+40].to_list()) for k in config['action'].modality_keys],axis=-1)
            predictions.append(pred); targets.append(target)
            for h in [1,5,10,20,40]:
                x,y=pred[h-1],target[h-1]; px,py=poses(x),poses(y); err=np.abs(x-y)
                r={'split':split,'episode':ei,'source_episode':entry['source_episode'],'frame':int(t),'horizon':h,'left_joint_mae_rad':float(err[:6].mean()),'right_joint_mae_rad':float(err[7:13].mean()),'left_gripper_mae':float(err[6]),'right_gripper_mae':float(err[13])}
                for i,side in enumerate(['left','right']):
                    r[f'{side}_tcp_error_m']=float(np.linalg.norm(px[i][:3,3]-py[i][:3,3])); r[f'{side}_orientation_error_rad']=float(Rotation.from_matrix(px[i][:3,:3].T@py[i][:3,:3]).magnitude())
                rows.append(r)
        print(json.dumps({'split':split,'episode':ei,'rows':len(rows),'seconds':time.time()-start}),flush=True)
        with (root/f'results/offline_action_metrics{a.output_suffix}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
np.savez_compressed(root/f'evaluation/offline_recovery_predictions{a.output_suffix}.npz',predictions=np.asarray(predictions),targets=np.asarray(targets))
summary=[]
for split in ['train','validation']:
    for h in [1,5,10,20,40]:
        rs=[r for r in rows if r['split']==split and r['horizon']==h]; summary.append({'split':split,'horizon':h,'samples':len(rs),**{k:float(np.mean([r[k] for r in rs])) for k in rows[0] if 'error' in k or 'mae' in k}})
(root/f'evaluation/offline_recovery_summary{a.output_suffix}.json').write_text(json.dumps({'sampling':'5 evenly spaced valid full-chunk anchors per episode; all 40 train and 5 validation episodes; not exhaustive frame evaluation','checkpoint':str((root/a.checkpoint).resolve()),'dataset_version':a.dataset_version,'denoising_steps':a.denoising_steps,'seconds':time.time()-start,'metrics':summary},indent=2))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(13,4))
for split in ['train','validation']:
    rs=[r for r in summary if r['split']==split]
    for ax,key in zip(axes,['joint_mae_rad','tcp_error_m','orientation_error_rad']):
        for side in ['left','right']: ax.plot([r['horizon'] for r in rs],[r[f'{side}_{key}'] for r in rs],marker='o',label=f'{split} {side}')
        ax.set(xlabel='Prediction step',ylabel=key); ax.legend(fontsize=7); ax.grid()
fig.tight_layout(); fig.savefig(root/f'results/offline_action_error_curves{a.output_suffix}.png',dpi=160)
