"""Test checkpoint's saved action processor; never changes weights/statistics."""
import json,runpy,argparse
from pathlib import Path
import numpy as np
from gr00t.policy.gr00t_policy import Gr00tPolicy
from transformers import AutoProcessor
import gr00t.model
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',default='checkpoints/lift_pot_2000/checkpoint-2000');ap.add_argument('--dataset',default='datasets/lerobot/lift_pot_train_v21');ap.add_argument('--output-suffix',default='');a=ap.parse_args()
root=Path(__file__).resolve().parents[1];cfg=runpy.run_path(str(root/'configs/aloha_gr00t_modality.py'))['config']; ep=LeRobotEpisodeLoader(root/a.dataset,cfg)[20]
p=AutoProcessor.from_pretrained(root/a.checkpoint);p.eval(); sap=p.state_action_processor; tag='new_embodiment'
print('tags',list(p.modality_configs),flush=True)
tag=next(t for t in p.modality_configs if t.lower()=='new_embodiment' or t=='31')
state={k:np.stack(ep[f'state.{k}'].iloc[0:1].to_list()) for k in cfg['state'].modality_keys}; action={k:np.stack(ep[f'action.{k}'].iloc[:40].to_list()) for k in cfg['action'].modality_keys}
norm=sap.apply_action(action,tag,state=state); decoded=sap.unapply_action(norm,tag,state=state)
r={'tag':tag,'sample':'train episode 20, first 40 actions','max_roundtrip_error':{k:float(np.max(np.abs(action[k]-decoded[k]))) for k in action},'relative_joints_absolute_grippers':True,'use_relative_action':p.use_relative_action,'use_percentiles':p.use_percentiles,'use_mean_std':p.use_mean_std,'clip_outliers':p.clip_outliers}
(root/f'evaluation/processor_roundtrip{a.output_suffix}.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
