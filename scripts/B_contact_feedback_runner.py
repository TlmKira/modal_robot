"""Paired real B-model ablations and separately labelled expert diagnostics."""
import argparse,json,os,sys,time,traceback,hashlib
from pathlib import Path
from multiprocessing.connection import Client
import numpy as np
import h5py
import yaml,sapien,imageio.v2 as imageio
from recovery_control import execute_fixed
from recovery_fk import poses
from gr00t_robotwin_interface import observation_for_policy,action_chunk_for_robot
root=Path(__file__).resolve().parents[1];repo=root/'simulation/RoboTwin';os.chdir(repo);sys.path.insert(0,str(repo))
from envs.lift_pot import lift_pot
ap=argparse.ArgumentParser();ap.add_argument('--run-label',default='');ap.add_argument('--port',type=int,default=6009);ap.add_argument('--seeds',type=int,nargs='+',default=[48,19,107]);a=ap.parse_args()
cp=(root/'checkpoints/lift_pot_rgb_B/checkpoint-2000').resolve();label_suffix=('_'+a.run_label) if a.run_label else '';out=root/f'evaluation/B_contact_feedback{label_suffix}';out.mkdir(exist_ok=True)
assert not list(out.glob('*_seed*.json')) and not list(out.glob('expert_actions_seed*.npz')), 'Use a fresh --run-label; preserving existing runs'
config=yaml.safe_load((repo/'env_cfg/task_config/demo_clean.yml').read_text());rf=repo/'assets/embodiments/aloha-agilex';rc=yaml.safe_load((rf/'config.yml').read_text())
config.update(task_name='lift_pot',task_config='demo_clean',embodiment_name='aloha-agilex',left_robot_file=str(rf),right_robot_file=str(rf),left_embodiment_config=rc,right_embodiment_config=rc,dual_arm_embodied=True,need_plan=True,save_data=False,save_path=str(root/'datasets/local_contact_diagnostic'),render_freq=0,save_freq=15,eval_mode=True,eval_video_save_dir=None)
stats=json.loads((cp/'statistics.json').read_text())['new_embodiment']['state'];keys=['left_arm','left_gripper','right_arm','right_gripper'];lo=np.concatenate([stats[k]['q01'] for k in keys]);hi=np.concatenate([stats[k]['q99'] for k in keys]);armidx=[0,1,2,3,4,5,7,8,9,10,11,12]
config_hash=hashlib.sha256((cp/'config.json').read_bytes()).hexdigest()
def make_task(seed):
 t=lift_pot();t.setup_demo(seed=seed,now_ep_num=seed,**config);assert int(t.model_id)==0,'This study is base0 only';return t

def real_gripper(robot,side):
 entity=getattr(robot,side+'_entity');joints=getattr(robot,side+'_gripper');active=entity.get_active_joints();q=entity.get_qpos();low,high=getattr(robot,side+'_gripper_scale');values=[]
 for joint,mult,offset in joints:
  scaled=(float(q[active.index(joint)])-offset)/mult;values.append((scaled-low)/(high-low))
 return values

def snapshot(t,tick,contact):
 drive=np.asarray(t.robot.get_left_arm_jointState()+t.robot.get_right_arm_jointState());real=np.asarray(t.robot.get_left_arm_real_jointState()+t.robot.get_right_arm_real_jointState());grip={s:real_gripper(t.robot,s) for s in ['left','right']};real[6]=grip['left'][0];real[13]=grip['right'][0]
 actual=[np.asarray(t.robot.get_left_tcp_pose()[:3]),np.asarray(t.robot.get_right_tcp_pose()[:3])];pd,pr=poses(drive),poses(real)
 fkerror=max(float(np.linalg.norm(pr[i][:3,3]-actual[i])) for i in [0,1]);assert fkerror<1e-4,f'FK/world mismatch {fkerror}'
 pose=t.pot.get_pose();rot=pose.to_transformation_matrix()[:3,:3]
 r={'physics_tick':tick,'sim_time_s':tick*.004,'action_count':t.take_action_cnt,'pot_height_m':float(pose.p[2]),'pot_up_dot':float(rot[2,2]),'official_success':bool(t.check_success()),'drive':drive.tolist(),'real_qpos_normalized_grippers':real.tolist(),'fk_validation_max_m':fkerror}
 for i,s,offset in [(0,'left',0),(1,'right',7)]:
  contactpoint=np.asarray(t.pot.get_contact_point(i)[:3]);r.update({f'{s}_grasp_distance_m':float(np.linalg.norm(actual[i]-contactpoint)),f'{s}_drive_tcp_grasp_distance_m':float(np.linalg.norm(pd[i][:3,3]-contactpoint)),f'{s}_tcp_tracking_m':float(np.linalg.norm(pd[i][:3,3]-actual[i])),f'{s}_joint_mae_rad':float(np.abs(drive[offset:offset+6]-real[offset:offset+6]).mean()),f'{s}_joint_max_error_rad':float(np.abs(drive[offset:offset+6]-real[offset:offset+6]).max()),f'{s}_gripper_command':float(drive[offset+6]),f'{s}_gripper_drive_normalized':float((getattr(t.robot,s+'_gripper')[0][0].get_drive_target()[0]-getattr(t.robot,s+'_gripper_scale')[0])/(getattr(t.robot,s+'_gripper_scale')[1]-getattr(t.robot,s+'_gripper_scale')[0])),f'{s}_gripper_actual_normalized':float(grip[s][0]),f'{s}_finger_actual_normalized':grip[s],f'{s}_finger_pot_contact':contact.get(s,False),f'{s}_arm_pot_contact':contact.get(s+'_arm',False),f'{s}_contact_impulse_ns':contact.get(s+'_impulse',0.0)})
 return r

class Recorder:
 def __init__(self,t,seed,variant):
  self.t=t;self.tick=0;self.rows=[];self.first={};self.contact_ticks={'left':0,'right':0,'both':0};self.video=root/f'videos/B_contact_feedback{label_suffix}_{variant}_seed{seed}_raw.mp4';assert not self.video.exists();self.writer=imageio.get_writer(self.video,fps=250/15,codec='libx264',quality=8,macro_block_size=1)
  position=np.array([1.2,-1.6,1.55]);forward=np.array([0,-.25,.78])-position;forward/=np.linalg.norm(forward);left=np.cross([0,0,1],forward);left/=np.linalg.norm(left);mat=np.eye(4);mat[:3,:3]=np.stack([forward,left,np.cross(forward,left)],axis=1);mat[:3,3]=position
  self.camera=t.scene.add_camera('contact_presentation',960,720,np.deg2rad(52),.1,10);self.camera.entity.set_pose(sapien.Pose(mat));self.original=t.scene.step
  self.potnames={l.entity.name for l in t.pot.actor.get_links()};self.fingers={s:{j.child_link.entity.name for j,_,_ in getattr(t.robot,s+'_gripper')} for s in ['left','right']};self.arms={s:{j.child_link.entity.name for j in getattr(t.robot,s+'_arm_joints')}|self.fingers[s] for s in ['left','right']};assert not self.potnames & (self.arms['left']|self.arms['right'])
  self.rows.append(snapshot(t,0,{}));self.capture();t.scene.step=self.step
 def contacts(self):
  result={}
  for c in self.t.scene.get_contacts():
   n=[b.entity.name for b in c.bodies]
   if not any(x in self.potnames for x in n):continue
   other=n[1] if n[0] in self.potnames else n[0];impulse=sum(float(np.linalg.norm(p.impulse)) for p in c.points)
   if impulse<=1e-8:continue
   for s in ['left','right']:
    if other in self.arms[s]:result[s+'_arm']=True
    if other in self.fingers[s]:result[s]=True;result[s+'_impulse']=result.get(s+'_impulse',0)+impulse
  return result
 def capture(self):
  self.t._update_render();self.camera.take_picture();self.writer.append_data(np.clip(self.camera.get_picture('Color')[...,:3]*255,0,255).astype(np.uint8))
 def step(self):
  self.original();self.tick+=1;c=self.contacts();fresh=False
  for key in ['left','right']:
   if c.get(key):self.contact_ticks[key]+=1
  if c.get('left') and c.get('right'):self.contact_ticks['both']+=1
  for key in ['left','right','left_arm','right_arm']:
   if c.get(key) and key not in self.first:self.first[key]={'physics_tick':self.tick,'sim_time_s':self.tick*.004,'action_count':self.t.take_action_cnt};fresh=True
  if fresh or self.tick%15==0:self.rows.append(snapshot(self.t,self.tick,c))
  if self.tick%15==0:self.capture()
 def close(self):
  self.rows.append(snapshot(self.t,self.tick,self.contacts()));self.capture();self.writer.close()

def expert_actions(seed):
 manifest=json.loads((root/'configs/dataset_split_manifest.json').read_text())
 entry=next(e for e in manifest['splits']['test'] if e['source_seed']==seed)
 with h5py.File(root/f"datasets/lift_pot/aloha-agilex_clean_50/data/episode{entry['source_episode']}.hdf5") as f:
  source=f['joint_action/vector'][:]
 t=make_task(seed)
 initial={'pot_pose':t.pot.get_pose().to_transformation_matrix().tolist(),'model_id':int(t.model_id),'source_episode':entry['source_episode'],'source_split':'test','source_initial_drive':source[0].tolist()}
 assert np.max(np.abs(source[0]-np.asarray(t.robot.get_left_arm_jointState()+t.robot.get_right_arm_jointState())))<1e-6
 t.close_env();path=out/f'expert_actions_seed{seed}.npz';assert not path.exists();np.savez(path,actions=source[1:]);return source[1:],initial

conn=Client(('127.0.0.1',a.port),authkey=b'local-gr00t-simulation')
variants=[('expert_replay','expert',1),('drive_h5','drive',5),('qpos_arm_h5','qpos_arm',5),('qpos_full_h5','qpos_full',5),('drive_h1','drive',1)]
try:
 for seed in a.seeds:
  actions,initial=expert_actions(seed)
  for variant,mode,horizon in variants:
   resultpath=out/f'{variant}_seed{seed}.json';assert not resultpath.exists(),'Preserve prior ablation result'
   t=make_task(seed);assert np.max(np.abs(t.pot.get_pose().to_transformation_matrix()-np.asarray(initial['pot_pose'])))<1e-6
   rec=Recorder(t,seed,variant);result={'seed':seed,'variant':variant,'state_mode':mode,'execute_horizon':horizon,'checkpoint':str(cp) if mode!='expert' else None,'checkpoint_config_sha256':config_hash if mode!='expert' else None,'expert_diagnostic_only':mode=='expert','model_id':int(t.model_id),'initial':initial,'physics_steps_per_action':15,'dt_s':.004,'denoising_steps':4,'ensemble':1,'success':False,'video':str(rec.video),'clipped_state_components':0,'policy_state_components':0};predictions=[];executed=[];policy_states=[];inference=[]
   try:
    if mode!='expert':conn.send({'op':'reset','seed':seed});assert conn.recv()['ok']
    while t.take_action_cnt<200 and not t.eval_success:
     if mode=='expert':
      if t.take_action_cnt>=len(actions):break
      chunk=actions[t.take_action_cnt:t.take_action_cnt+1]
     else:
      obs=t.get_obs();latest=snapshot(t,rec.tick,rec.contacts());state=np.asarray(latest['drive']) if mode=='drive' else np.asarray(latest['real_qpos_normalized_grippers'])
      if mode=='qpos_arm':state[[6,13]]=np.asarray(latest['drive'])[[6,13]]
      if mode=='qpos_full':state[[6,13]]=np.clip(state[[6,13]],0,1)
      for key,sl in zip(keys,[slice(0,6),slice(6,7),slice(7,13),slice(13,14)]):obs['joint_action'][key]=state[sl]
      policy_states.append(state);result['clipped_state_components']+=int(np.count_nonzero((state<lo)|(state>hi)));result['policy_state_components']+=14
      started=time.perf_counter();conn.send({'op':'action','observation':observation_for_policy(obs,'Lift the pot with both arms.'),'denoising_steps':4,'ensemble':1});response=conn.recv()
      if 'error' in response:raise RuntimeError(response['error'])
      assert Path(response['checkpoint']).resolve()==cp
      inference.append({'sim_time_s':rec.tick*.004,'action_count':t.take_action_cnt,'model_seconds':response['inference_seconds'],'bridge_wall_seconds':time.perf_counter()-started})
      limits=np.asarray([j.get_limits()[0] for j in t.robot.left_arm_joints+t.robot.right_arm_joints]);chunk=action_chunk_for_robot(response['actions'],limits);predictions.append(chunk)
     for action in chunk[:horizon]:
      execute_fixed(t,action,15);executed.append(np.asarray(t.robot.get_left_arm_jointState()+t.robot.get_right_arm_jointState()))
      if t.eval_success or t.take_action_cnt>=200:break
     print(json.dumps({'seed':seed,'variant':variant,'actions':t.take_action_cnt,'success':bool(t.eval_success)}),flush=True)
     if t.pot.get_pose().p[2]<initial['pot_pose'][2][3]-.12:raise RuntimeError('Pot fell below table')
    result['success']=bool(t.check_success())
   except Exception:result['error']=traceback.format_exc();print(result['error'],flush=True)
   finally:
    rec.close();result.update(actions=t.take_action_cnt,physics_ticks=rec.tick,first_contact=rec.first,physical_finger_contact_ticks=rec.contact_ticks,telemetry=rec.rows,inference=inference,contact_definition='pot-link to finger/arm link with nonzero summed contact impulse >1e-8 Ns; table contact excluded',finger_link_names={s:sorted(v) for s,v in rec.fingers.items()},pot_link_names=sorted(rec.potnames));resultpath.write_text(json.dumps(result,indent=2));np.savez_compressed(out/f'{variant}_seed{seed}_actions.npz',predictions=np.asarray(predictions),executed=np.asarray(executed),policy_states=np.asarray(policy_states));t.close_env()
    print(json.dumps({k:v for k,v in result.items() if k not in ['telemetry','inference']}),flush=True)
finally:conn.close()
