"""Official expert -> fixed 60ms drive replay -> contact/collision/2s hold QC. Never trains."""
import os,sys,json,time,traceback,argparse,hashlib
from pathlib import Path
import numpy as np,h5py,yaml
ROOT=Path(__file__).resolve().parents[1];repo=ROOT/'simulation/RoboTwin';os.chdir(repo);sys.path.insert(0,str(repo))
from envs.lift_pot import lift_pot
p=argparse.ArgumentParser();p.add_argument('--label',default='qualified_v1');p.add_argument('--target',type=int,default=50);p.add_argument('--max-attempts',type=int,default=250);p.add_argument('--start-seed',type=int,default=320000);p.add_argument('--skip-candidates',action='store_true');a=p.parse_args();out=ROOT/'datasets'/f'lift_pot_C_{a.label}';out.mkdir(exist_ok=False)
cfg=yaml.safe_load((repo/'env_cfg/task_config/demo_clean.yml').read_text());rf=repo/'assets/embodiments/aloha-agilex';rc=yaml.safe_load((rf/'config.yml').read_text());cfg.update(task_name='lift_pot',task_config='demo_clean',embodiment_name='aloha-agilex',left_robot_file=str(rf),right_robot_file=str(rf),left_embodiment_config=rc,right_embodiment_config=rc,dual_arm_embodied=True,need_plan=True,save_data=False,save_path=str(out),render_freq=0,save_freq=15,eval_mode=False)
(out/'configuration.json').write_text(json.dumps({'config':cfg,'target':a.target,'seed_range':'310000-310011 candidates; 320000+ new training; blind 410000-410019 excluded','source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'state':'absolute drive positions left6/grip/right6/grip','sampling_s':.06,'hold_s':2,'collision_threshold_ns':.01},indent=2))
results=[];accepted=0
candidates={r['seed']:r['file'] for name in ['pilot01','pilot02'] for r in json.loads((ROOT/f'results/C_preparation/{name}_quality.json').read_text())['records'] if r['accepted']}
def setup(seed):
 t=lift_pot();t.setup_demo(seed=seed,now_ep_num=seed,**cfg);return t
def drive(t):return np.asarray(t.robot.get_left_arm_jointState()+t.robot.get_right_arm_jointState())
for seed in ([] if a.skip_candidates else list(candidates))+list(range(a.start_seed,a.start_seed+a.max_attempts)):
 if accepted>=a.target:break
 started=time.perf_counter();r={'seed':seed,'accepted':False,'source':'existing_candidate' if seed in candidates else 'official_expert_new'};t=None
 try:
  t=setup(seed);r['model_id']=int(t.model_id);r['initial_pose']=t.pot.get_pose().to_transformation_matrix().tolist()
  if t.model_id!=0:r['reason']='base1 excluded; separate planning issue';continue
  if seed in candidates:
   with h5py.File(candidates[seed]) as f:commands=f['state_drive'][:][f['physics_tick'][:]%15==0]
   r['original_official_success']=True;r['planned_legal']=True
  else:
   ticks=[0];commands=[drive(t)];original=t.scene.step
   def plan_step():
    original();ticks[0]+=1
    if ticks[0]%15==0:commands.append(drive(t))
   t.scene.step=plan_step;t.play_once();legal,excess=t.planned_joints_legal();r.update(original_official_success=bool(t.check_success()),planned_legal=bool(legal),plan_success=bool(t.plan_success))
   if not(t.plan_success and t.check_success() and legal):r['reason']='official expert planning/task/limits failed';continue
  commands=np.asarray(commands);np.savez_compressed(out/f'expert_commands_seed{seed}.npz',commands=commands)
  t.close_env();t=setup(seed);assert np.max(np.abs(t.pot.get_pose().to_transformation_matrix()-r['initial_pose']))<1e-6;assert np.max(np.abs(drive(t)-commands[0]))<1e-6
  potnames={l.entity.name for l in t.pot.actor.get_links()};fingers={s:{j.child_link.entity.name for j,_,_ in getattr(t.robot,s+'_gripper')} for s in ['left','right']};arms={s:{j.child_link.entity.name for j in getattr(t.robot,s+'_arm_joints')} for s in ['left','right']};robotnames=arms['left']|arms['right']|fingers['left']|fingers['right'];original=t.scene.step;tick=0;frames=[];images={k:[] for k in ['head_camera','left_camera','right_camera']};telemetry=[]
  def status():
   c=[False,False];bad=[]
   for contact in t.scene.get_contacts():
    n=[b.entity.name for b in contact.bodies];imp=sum(float(np.linalg.norm(point.impulse)) for point in contact.points)
    if imp<=1e-8:continue
    if any(v in potnames for v in n):
     other=n[1] if n[0] in potnames else n[0]
     for i,s in enumerate(['left','right']):
      if other in fingers[s]:c[i]=True
      if other in arms[s] and imp>.01:bad.append({'names':n,'impulse':imp})
    elif any(v in robotnames for v in n) and imp>.01:
     # Cross-arm or moving robot link against external scene; same-arm selfcontacts logged as collision too.
     if n[0]!=n[1]:bad.append({'names':n,'impulse':imp})
   pos=t.pot.get_pose();real=np.asarray(t.robot.get_left_arm_real_jointState()+t.robot.get_right_arm_real_jointState());q=drive(t);dist=[np.linalg.norm(np.asarray(t.robot.get_left_tcp_pose()[:3])-t.pot.get_contact_point(0)[:3]),np.linalg.norm(np.asarray(t.robot.get_right_tcp_pose()[:3])-t.pot.get_contact_point(1)[:3])]
   physical=[]
   for s in ['left','right']:
    ent=getattr(t.robot,s+'_entity');js=ent.get_active_joints();physical.extend(float(ent.get_qpos()[js.index(j)]) for j,_,_ in getattr(t.robot,s+'_gripper'))
   return {'tick':tick,'official':bool(t.check_success()),'contacts':c,'collisions':bad,'height':float(pos.p[2]),'up_dot':float(pos.to_transformation_matrix()[2,2]),'grasp_distance':np.asarray(dist).tolist(),'drive':q.tolist(),'real_arm_command_gripper':real.tolist(),'finger_m':physical}
  def capture(row):
   obs=t.get_obs();frames.append(row)
   for k in images:images[k].append(obs['observation'][k]['rgb'].copy())
  initial=status();capture(initial)
  def step():
   nonlocal_dummy=None
   global tick
   original();tick+=1;row=status();telemetry.append(row)
   if tick%15==0:capture(row)
  t.scene.step=step
  for command in commands[1:]:
   current=drive(t);velocity=(command-current)/.06
   for k in range(1,16):
    q=current+(command-current)*k/15;t.robot.set_arm_joints(q[:6],velocity[:6],'left');t.robot.set_arm_joints(q[7:13],velocity[7:13],'right');t.robot.set_gripper(float(q[6]),'left');t.robot.set_gripper(float(q[13]),'right');t.scene.step()
  r['replay_official_success']=bool(t.check_success());hold_start=tick;final=drive(t)
  for k in range(510): # 2.04s, divisible by 15; require first2s.
   t.robot.set_arm_joints(final[:6],np.zeros(6),'left');t.robot.set_arm_joints(final[7:13],np.zeros(6),'right');t.robot.set_gripper(float(final[6]),'left');t.robot.set_gripper(float(final[13]),'right');t.scene.step()
  hold=[x for x in telemetry if hold_start<x['tick']<=hold_start+500];both=sum(all(x['contacts']) for x in hold)*.004;collisions=sum(bool(x['collisions']) for x in telemetry);states=np.asarray([x['drive'] for x in frames]);r.update(hold_official_all=all(x['official'] for x in hold),hold_both_contact_s=both,collision_ticks=collisions,frames=len(frames),physics_ticks=tick,gripper_closed_final=bool(np.max(states[-1,[6,13]])<.1),final=telemetry[-1],first_contact={s:next((x for x in telemetry if x['contacts'][i]),None) for i,s in enumerate(['left','right'])})
  r['accepted']=bool(r['replay_official_success'] and r['hold_official_all'] and both>=.8 and collisions==0 and r['gripper_closed_final'] and np.isfinite(states).all())
  r['reason']='qualified' if r['accepted'] else 'replay/hold/contact/collision/gripper QC failed'
  # Keep full diagnostics even when rejected; only accepted manifests feed training.
  (out/f'telemetry_seed{seed}.json').write_text(json.dumps(telemetry))
  path=out/f"{'accepted' if r['accepted'] else 'rejected'}_seed{seed}.hdf5"
  with h5py.File(path,'w') as f:
   f.attrs.update(seed=seed,model_id=0,rgb_order='RGB',state_semantics='drive',official_success=r['replay_official_success'],quality_accepted=r['accepted'])
   f.create_dataset('state_drive',data=states);f.create_dataset('physics_tick',data=[x['tick'] for x in frames]);f.create_dataset('timestamp_s',data=np.arange(len(frames))*.06);f.create_dataset('real_arm_command_gripper',data=[x['real_arm_command_gripper'] for x in frames]);f.create_dataset('physical_finger_qpos_m',data=[x['finger_m'] for x in frames]);f.create_dataset('grasp_distance_m',data=[x['grasp_distance'] for x in frames]);f.create_dataset('contacts',data=[x['contacts'] for x in frames]);f.create_dataset('pot_height',data=[x['height'] for x in frames]);f.create_dataset('pot_up_dot',data=[x['up_dot'] for x in frames])
   for k,v in images.items():f.create_dataset('rgb/'+k,data=np.asarray(v),compression='gzip',compression_opts=1)
  r['file']=str(path);r['bytes']=path.stat().st_size
  if r['accepted']:accepted+=1
 except Exception:r['reason']='exception';r['error']=traceback.format_exc()
 finally:
  r['wall_s']=time.perf_counter()-started;results.append(r);(out/'manifest.json').write_text(json.dumps({'accepted':accepted,'target':a.target,'records':results},indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['final','first_contact']}),flush=True)
  if t is not None and hasattr(t,'scene'):t.close_env()
