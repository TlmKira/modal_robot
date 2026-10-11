"""Frozen drive_h5 paired blind evaluation; continue model control for 2s after success."""
import argparse,json,os,sys,time,traceback,hashlib
from pathlib import Path
from multiprocessing.connection import Client
import numpy as np,yaml
from BC_blind_telemetry import Recorder,snapshot
from control_audit import validate_action_timing, summarize_hold, SuccessEvents
from gr00t_robotwin_interface import observation_for_policy,action_chunk_for_robot
root=Path(__file__).resolve().parents[1];repo=root/'simulation/RoboTwin'
ap=argparse.ArgumentParser();ap.add_argument('--run-label',required=True);ap.add_argument('--config-path',default='configs/C_blind_evaluation_base0_v2.json');ap.add_argument('--model',choices=['B','C'],required=True);ap.add_argument('--port',type=int,default=6011);a=ap.parse_args()
os.chdir(repo);sys.path.insert(0,str(repo))
from envs.lift_pot import lift_pot
assert a.run_label.replace('_','').replace('-','').isalnum(), 'Use a simple unique run label'
cfgpath=root/a.config_path
cfg=json.loads(cfgpath.read_text());cp=(root/cfg['baseline_B' if a.model=='B' else 'C']).resolve()
out=root/f'evaluation/BC_blind_{a.run_label}';out.mkdir(exist_ok=True)
config=yaml.safe_load((repo/'env_cfg/task_config/demo_clean.yml').read_text());rf=repo/'assets/embodiments/aloha-agilex';rc=yaml.safe_load((rf/'config.yml').read_text())
config.update(task_name='lift_pot',task_config='demo_clean',embodiment_name='aloha-agilex',left_robot_file=str(rf),right_robot_file=str(rf),left_embodiment_config=rc,right_embodiment_config=rc,dual_arm_embodied=True,need_plan=True,save_data=False,save_path=str(root/'datasets/local_blind_rollouts'),render_freq=0,save_freq=15,eval_mode=True,eval_video_save_dir=None)
conn=Client(('127.0.0.1',a.port),authkey=b'local-gr00t-simulation')
def execute(t,action,rec,success):
 validate_action_timing(t.scene.get_timestep())
 current=np.asarray(t.robot.get_left_arm_jointState()+t.robot.get_right_arm_jointState());velocity=(action-current)/.06;t.take_action_cnt+=1
 for i in range(1,16):
  target=current+(action-current)*i/15
  t.robot.set_arm_joints(target[:6],velocity[:6],'left');t.robot.set_arm_joints(target[7:13],velocity[7:13],'right')
  t.robot.set_gripper(float(target[6]),'left');t.robot.set_gripper(float(target[13]),'right');t.scene.step()
  official=bool(t.check_success())
  for event in success['events'].observe(rec.tick,official):rec.record_event(event,official)
  if official and success['first_tick'] is None:success['first_tick']=rec.tick
  if success['first_tick'] is not None:success['hold_ticks'].append(dict(tick=rec.tick,official=official,both_contact=bool(rec.contacts().get('left') and rec.contacts().get('right'))))
try:
 for seed in cfg['seeds']:
  path=out/f'{a.model}_seed{seed}.json'
  assert not path.exists(),'Never repeat an already evaluated blind seed'
  t=lift_pot()
  try:
   t.setup_demo(seed=seed,now_ep_num=seed,**config);assert int(t.model_id)==0
   assert t.take_action_cnt == 0, 'Episode action counter did not reset'
   validate_action_timing(t.scene.get_timestep())
   initial_drive=np.asarray(t.robot.get_left_arm_jointState()+t.robot.get_right_arm_jointState())
   if initial_drive.shape != (14,) or not np.isfinite(initial_drive).all():raise ValueError('Invalid initial drive state')
   rec=Recorder(t,seed,a.model,a.run_label);predictions=[];executed=[];states=[];latencies=[];success={'first_tick':None,'hold_ticks':[],'events':SuccessEvents()}
   r=dict(model=a.model,seed=seed,checkpoint=str(cp),model_id=0,expert_called=False,controller='drive_h5',physics_dt_s=.004,ticks_per_action=15,denoising_steps=4,ensemble=1,video=str(rec.video),configuration_sha256=hashlib.sha256(cfgpath.read_bytes()).hexdigest(),expert_check=False,diagnostic_after_blind_test=True,record_schema_version=3,initial_pot_matrix=t.pot.get_pose().to_transformation_matrix().tolist())
   try:
    conn.send({'op':'reset','seed':seed});assert conn.recv()['ok']
    while (success['first_tick'] is None and t.take_action_cnt<200) or (success['first_tick'] is not None and rec.tick-success['first_tick']<500):
     obs=t.get_obs();state=np.asarray(t.robot.get_left_arm_jointState()+t.robot.get_right_arm_jointState());states.append(state)
     for key,sl in zip(['left_arm','left_gripper','right_arm','right_gripper'],[slice(0,6),slice(6,7),slice(7,13),slice(13,14)]):obs['joint_action'][key]=state[sl]
     started=time.perf_counter();conn.send({'op':'action','observation':observation_for_policy(obs,'Lift the pot with both arms.'),'denoising_steps':4,'ensemble':1});response=conn.recv()
     if 'error' in response:raise RuntimeError(response['error'])
     assert Path(response['checkpoint']).resolve()==cp
     latencies.append(dict(action_count=t.take_action_cnt,sim_time_s=rec.tick*.004,model_seconds=response['inference_seconds'],bridge_seconds=time.perf_counter()-started))
     limits=np.asarray([j.get_limits()[0] for j in t.robot.left_arm_joints+t.robot.right_arm_joints]);chunk=action_chunk_for_robot(response['actions'],limits);predictions.append(chunk)
     for action in chunk[:5]:
      execute(t,action,rec,success);executed.append(action)
      if success['first_tick'] is not None and rec.tick-success['first_tick']>=500:break
      if success['first_tick'] is None and t.take_action_cnt>=200:break
     if t.pot.get_pose().p[2]<r['initial_pot_matrix'][2][3]-.12:
      r['pot_fell_below_table']=True;break
   except Exception:r['error']=traceback.format_exc();print(r['error'],flush=True)
   finally:
    rec.close();first=success['first_tick'];hold=success['hold_ticks'];holding=summarize_hold(first,rec.tick,hold)
    r.update(official_success=first is not None,first_success_tick=first,hold_2s_completed=holding['completed'],hold_2s_continuous_official=holding['continuous_official'],hold_2s_both_contact_seconds=holding['both_contact_seconds'],actions=t.take_action_cnt,physics_ticks=rec.tick,first_contact=rec.first,contact_ticks=rec.contact_ticks,telemetry=rec.rows,success_events=rec.events,inference=latencies,hold_telemetry=hold,holding_protocol='Same model replans every 5 actions for 2 seconds after first official success; no target freeze. Official result latched independently.',simulation_paused_during_inference=True)
    path.write_text(json.dumps(r,indent=2));np.savez_compressed(out/f'{a.model}_seed{seed}_actions.npz',predictions=np.asarray(predictions),executed=np.asarray(executed),policy_states=np.asarray(states))
    print(json.dumps(dict(model=a.model,seed=seed,official_success=r['official_success'],hold_2s=r['hold_2s_continuous_official'],actions=r['actions'])),flush=True)
  finally:
   t.close_env()
finally:conn.close()
