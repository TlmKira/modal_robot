"""Check actual logged sampler truncation against each original expert trajectory."""
from pathlib import Path
import h5py,numpy as np,json
root=Path(__file__).resolve().parents[1];m=json.loads((root/'configs/dataset_split_manifest.json').read_text());rows=[]
for e in m['splits']['train']:
 with h5py.File(root/f"datasets/lift_pot/aloha-agilex_clean_50/data/episode{e['source_episode']}.hdf5") as f:q=f['joint_action/vector'][:]
 n=e['frames'];valid=n-39;closed=(q[:n,6]<.1)&(q[:n,13]<.1)
 rows.append({**e,'sampled_observations':valid,'excluded_observations':39,'sampled_state_gripper_min':q[:valid,[6,13]].min(0).tolist(),'sampled_first_action_gripper_min':q[1:valid+1,[6,13]].min(0).tolist(),'real_closed_gripper_states':int(closed.sum()),'sampled_closed_gripper_states':int(closed[:valid].sum()),'excluded_closed_gripper_states':int(closed[valid:].sum())})
r={'action_horizon':40,'original_real_frames':sum(x['frames'] for x in rows),'actual_training_anchor_count':sum(x['sampled_observations'] for x in rows),'excluded_anchor_count':sum(x['excluded_observations'] for x in rows),'closed_states_real':sum(x['real_closed_gripper_states'] for x in rows),'closed_states_sampled':sum(x['sampled_closed_gripper_states'] for x in rows),'sampled_state_gripper_min':np.min([x['sampled_state_gripper_min'] for x in rows],axis=0).tolist(),'sampled_first_action_gripper_min':np.min([x['sampled_first_action_gripper_min'] for x in rows],axis=0).tolist(),'rows':rows,'evidence':'sharded_single_step_dataset.py get_effective_episode_length; both training logs report Total steps: 2884; allow_padding=false'}
(root/'evaluation/training_tail_coverage_audit.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k!='rows'})
