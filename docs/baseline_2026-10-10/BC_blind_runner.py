"""Frozen drive_h5 paired blind evaluation; continue model control for 2s after success."""
import argparse,json,os,sys,time,traceback,hashlib
from pathlib import Path
from multiprocessing.connection import Client
import numpy as np,yaml
from BC_blind_telemetry import Recorder,snapshot
from gr00t_robotwin_interface import observation_for_policy,action_chunk_for_robot
root=Path(__file__).resolve().parents[1];repo=root/'simulation/RoboTwin'
ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['B','C'],required=True);ap.add_argument('--port',type=int,default=6011);a=ap.parse_args()
os.chdir(repo);sys.path.insert(0,str(repo))
from envs.lift_pot import lift_pot
cfg=json.loads((root/'configs/C_blind_evaluation_base0_v2.json').read_text());cp=(root/cfg['baseline_B' if a.model=='B' else 'C']).resolve()
out=root/'evaluation/BC_blind_v2';out.mkdir(exist_ok=True)
config=yaml.safe_load((repo/'env_cfg/task_config/demo_clean.yml').read_text());rf=repo/'assets/embodiments/aloha-agilex';rc=yaml.safe_load((rf/'config.yml').read_text())
config.update(task_name='lift_pot',task_config='demo_clean',embodiment_name='aloha-agilex',left_robot_file=str(rf),right_robot_file=str(rf),left_embodiment_config=rc,right_embodiment_config=rc,dual_arm_embodied=True,need_plan=True,save_data=False,save_path=str(root/'datasets/local_blind_rollouts'),render_freq=0,save_freq=15,eval_mode=True,eval_video_save_dir=None)
conn=Client(('127.0.0.1',a.port),authkey=b'local-gr00t-simulation')
def execute(t,action,rec,success):
 current=np.asarray(t.robot.get_left_arm_jointState()+t.robot.get_right_arm_jointState());velocity=(action-current)/.06;t.take_action_cnt+=1
 for i in range(1,16):
  target=current+(action-current)*i/15
  t.robot.set_arm_joints(target[:6],velocity[:6],'left');t.robot.set_arm_joints(target[7:13],velocity[7:13],'right')
  t.robot.set_gripper(float(target[6]),'left');t.robot.set_gripper(float(target[13]),'right');t.scene.step()
  official=bool(t.check_success())
  if official and success['first_tick'] is None:success['first_tick']=rec.tick
  if success['first_tick'] is not None:success['hold_ticks'].append(dict(tick=rec.tick,official=official,both_contact=bool(rec.contacts().get('left') and rec.contacts().get('right'))))
try:
 for seed in cfg['seeds']:
  path=out/f'{a.model}_seed{seed}.json'
  assert not path.exists(),'Never repeat an already evaluated blind seed'
  t=lift_pot();t.setup_demo(seed=seed,now_ep_num=seed,**config);assert int(t.model_id)==0
  rec=Recorder(t,seed,a.model);predictions=[];executed=[];states=[];latencies=[];success={'first_tick':None,'hold_ticks':[]}
  r=dict(model=a.model,seed=seed,checkpoint=str(cp),model_id=0,expert_called=False,controller='drive_h5',physics_dt_s=.004,ticks_per_action=15,denoising_steps=4,ensemble=1,video=str(rec.video),configuration_sha256=hashlib.sha256((root/'configs/C_blind_evaluation_base0_v2.json').read_bytes()).hexdigest(),initial_pot_matrix=t.pot.get_pose().to_transformation_matrix().tolist())
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
   rec.close();first=success['first_tick'];hold=[h for h in success['hold_ticks'] if first is not None and first<=h['tick']<=first+500]
   r.update(official_success=first is not None,first_success_tick=first,hold_2s_completed=first is not None and rec.tick-first>=500,hold_2s_continuous_official=first is not None and rec.tick-first>=500 and all(h['official'] for h in hold),hold_2s_both_contact_seconds=sum(h['both_contact'] for h in hold)*.004,actions=t.take_action_cnt,physics_ticks=rec.tick,first_contact=rec.first,contact_ticks=rec.contact_ticks,telemetry=rec.rows,inference=latencies,hold_telemetry=hold,holding_protocol='Same model replans every 5 actions for 2 seconds after first official success; no target freeze. Official result latched independently.',simulation_paused_during_inference=True)
   path.write_text(json.dumps(r,indent=2));np.savez_compressed(out/f'{a.model}_seed{seed}_actions.npz',predictions=np.asarray(predictions),executed=np.asarray(executed),policy_states=np.asarray(states));t.close_env()
   print(json.dumps(dict(model=a.model,seed=seed,official_success=r['official_success'],hold_2s=r['hold_2s_continuous_official'],actions=r['actions'])),flush=True)
finally:conn.close()
