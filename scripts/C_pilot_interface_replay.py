"""Independent stability diagnostic: replay saved B commands, then hold final targets. No new inference."""
import json,os,sys
from pathlib import Path
import numpy as np,yaml
from recovery_control import execute_fixed
root=Path(__file__).resolve().parents[1];repo=root/'simulation/RoboTwin';os.chdir(repo);sys.path.insert(0,str(repo))
from envs.lift_pot import lift_pot
out=root/'evaluation/C_pilot_interface_replay';out.mkdir(exist_ok=False)
cfg=yaml.safe_load((repo/'env_cfg/task_config/demo_clean.yml').read_text());rf=repo/'assets/embodiments/aloha-agilex';rc=yaml.safe_load((rf/'config.yml').read_text());cfg.update(task_name='lift_pot',task_config='demo_clean',embodiment_name='aloha-agilex',left_robot_file=str(rf),right_robot_file=str(rf),left_embodiment_config=rc,right_embodiment_config=rc,dual_arm_embodied=True,need_plan=True,save_data=False,save_path=str(out),render_freq=0,save_freq=15,eval_mode=True,eval_video_save_dir=None)
import h5py
seed=310005
t=lift_pot();t.setup_demo(seed=seed,now_ep_num=seed,**cfg)
with h5py.File(root/'datasets/lift_pot_C_pilot01/rejected_seed310005.hdf5') as f:
 commands=f['state_drive'][:];ticks=f['physics_tick'][:]
# Keep initial and all global 15-tick samples. Do not execute an irregular final frame as a full action.
commands=commands[ticks%15==0]
for command in commands[1:]:execute_fixed(t,command,15)
result={'seed':seed,'method':'new official expert sampled drive targets through B fixed 15x4ms executor','official_success':bool(t.check_success()),'pot_position':t.pot.get_pose().p.tolist(),'actions':t.take_action_cnt,'irregular_tail_excluded':True}
(out/'expert_seed310005.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));t.close_env()
