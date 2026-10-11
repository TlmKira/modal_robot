"""Select official base0 initializations before any B/C rollout, without expert calls."""
import json,os,sys,time
from pathlib import Path
import yaml
root=Path(__file__).resolve().parents[1];repo=root/'simulation/RoboTwin'
os.chdir(repo);sys.path.insert(0,str(repo))
from envs.lift_pot import lift_pot
config=yaml.safe_load((repo/'env_cfg/task_config/demo_clean.yml').read_text())
rf=repo/'assets/embodiments/aloha-agilex';rc=yaml.safe_load((rf/'config.yml').read_text())
config.update(task_name='lift_pot',task_config='demo_clean',embodiment_name='aloha-agilex',left_robot_file=str(rf),right_robot_file=str(rf),left_embodiment_config=rc,right_embodiment_config=rc,dual_arm_embodied=True,need_plan=True,save_data=False,save_path=str(root/'datasets/local_blind_seed_setup'),render_freq=0,save_freq=15,eval_mode=True,eval_video_save_dir=None)
path=root/'evaluation/C_blind_base0_seed_selection.json'
assert not path.exists(),'Preserve previous frozen selection'
rows=[];selected=[]
for seed in range(410000,410300):
 t=lift_pot()
 try:
  t.setup_demo(seed=seed,now_ep_num=seed,**config)
  model_id=int(t.model_id)
  rows.append(dict(seed=seed,model_id=model_id,initial_pot_matrix=t.pot.get_pose().to_transformation_matrix().tolist()))
  if model_id==0:selected.append(seed)
 finally:t.close_env()
 result=dict(status='frozen' if len(selected)==20 else 'selecting',selected_seeds=selected,candidates=rows,selection='First 20 base0 seeds in ascending order from 410000; official setup only; no policy inference or expert planning',expert_called=False,model_called=False)
 path.write_text(json.dumps(result,indent=2))
 print(json.dumps(dict(seed=seed,model_id=model_id,selected=len(selected))),flush=True)
 if len(selected)==20:
  old=json.loads((root/'configs/C_blind_evaluation_v1.json').read_text())
  old.update(status='base0_seeds_frozen_before_B_C_inference',previous_unfiltered_reservation=old['seeds'],seeds=selected,sampling=result['selection'],C='checkpoints/lift_pot_C_v1/checkpoint-2000',seed_selection_record=str(path.relative_to(root)))
  (root/'configs/C_blind_evaluation_base0_v2.json').write_text(json.dumps(old,indent=2))
  break
assert len(selected)==20
