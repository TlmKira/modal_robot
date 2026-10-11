"""Independent stability diagnostic: replay saved B commands, then hold final targets. No new inference."""
import json,os,sys
from pathlib import Path
import numpy as np,yaml
from recovery_control import execute_fixed
root=Path(__file__).resolve().parents[1];repo=root/'simulation/RoboTwin';os.chdir(repo);sys.path.insert(0,str(repo))
from envs.lift_pot import lift_pot
out=root/'evaluation/B_saved_action_hold';out.mkdir(exist_ok=False)
cfg=yaml.safe_load((repo/'env_cfg/task_config/demo_clean.yml').read_text());rf=repo/'assets/embodiments/aloha-agilex';rc=yaml.safe_load((rf/'config.yml').read_text());cfg.update(task_name='lift_pot',task_config='demo_clean',embodiment_name='aloha-agilex',left_robot_file=str(rf),right_robot_file=str(rf),left_embodiment_config=rc,right_embodiment_config=rc,dual_arm_embodied=True,need_plan=True,save_data=False,save_path=str(out),render_freq=0,save_freq=15,eval_mode=True,eval_video_save_dir=None)
for seed,variant in [(19,'qpos_full_h5'),(107,'drive_h5')]:
 t=lift_pot();t.setup_demo(seed=seed,now_ep_num=seed,**cfg);data=np.load(root/f'evaluation/B_contact_feedback/{variant}_seed{seed}_actions.npz');commands=data['executed'];rows=[]
 for command in commands:execute_fixed(t,command,15)
 initial_success=bool(t.check_success());target=np.array(t.robot.get_left_arm_jointState()+t.robot.get_right_arm_jointState());rows.append({'time_s':0.,'success':initial_success,'pot_position':t.pot.get_pose().p.tolist()})
 for i in range(500):
  t.robot.set_arm_joints(target[:6],np.zeros(6),'left');t.robot.set_arm_joints(target[7:13],np.zeros(6),'right');t.robot.set_gripper(float(target[6]),'left');t.robot.set_gripper(float(target[13]),'right');t.scene.step()
  if (i+1)%15==0 or i==499:rows.append({'time_s':(i+1)*.004,'success':bool(t.check_success()),'pot_position':t.pot.get_pose().p.tolist()})
 result={'seed':seed,'variant':variant,'method':'saved GR00T executed command replay; no live model inference; fixed final drive target for two seconds','replay_reproduced_official_success':initial_success,'hold_sampled_official_success_all':initial_success and all(r['success'] for r in rows),'hold_final_success':rows[-1]['success'],'telemetry':rows,'original_official_result_unchanged':True};(out/f'{variant}_seed{seed}.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='telemetry'}),flush=True);t.close_env()
