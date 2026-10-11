"""Non-destructive official expert pilot; RGB and drive/physical states at exact physics ticks."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
import numpy as np,h5py,yaml
ROOT=Path(__file__).resolve().parents[1];repo=ROOT/'simulation/RoboTwin'
os.chdir(repo);sys.path.insert(0,str(repo))
from envs.lift_pot import lift_pot
p=argparse.ArgumentParser();p.add_argument('--start-seed',type=int,default=310000);p.add_argument('--attempts',type=int,default=6);p.add_argument('--label',default='pilot01');a=p.parse_args()
out=ROOT/'datasets'/f'lift_pot_C_{a.label}';out.mkdir(exist_ok=False)
cfg=yaml.safe_load((repo/'env_cfg/task_config/demo_clean.yml').read_text());rf=repo/'assets/embodiments/aloha-agilex';rc=yaml.safe_load((rf/'config.yml').read_text())
cfg.update(task_name='lift_pot',task_config='demo_clean',embodiment_name='aloha-agilex',left_robot_file=str(rf),right_robot_file=str(rf),left_embodiment_config=rc,right_embodiment_config=rc,dual_arm_embodied=True,need_plan=True,save_data=False,save_path=str(out),render_freq=0,save_freq=15,eval_mode=False)
results=[]
for seed in range(a.start_seed,a.start_seed+a.attempts):
 t=lift_pot();started=time.perf_counter();r={'seed':seed,'accepted':False};rows=[];rgb={};contact_rows=[];tcp_rows=[];tick=[0]
 try:
  t.setup_demo(seed=seed,now_ep_num=seed,**cfg);r.update(model_id=int(t.model_id),initial_pot_pose=t.pot.get_pose().to_transformation_matrix().tolist());original=t.scene.step
  def capture():
   obs=t.get_obs();drive=np.array(t.robot.get_left_arm_jointState()+t.robot.get_right_arm_jointState());real=np.array(t.robot.get_left_arm_real_jointState()+t.robot.get_right_arm_real_jointState());physical=[]
   for side in ['left','right']:
    ent=getattr(t.robot,side+'_entity');active=ent.get_active_joints();physical.extend(float(ent.get_qpos()[active.index(j)]) for j,_,_ in getattr(t.robot,side+'_gripper'))
   rows.append((tick[0],drive,real,physical,t.pot.get_pose().p.copy(),bool(t.check_success())))
   potnames={l.entity.name for l in t.pot.actor.get_links()};fingers={side:{j.child_link.entity.name for j,_,_ in getattr(t.robot,side+'_gripper')} for side in ['left','right']};contact=[False,False]
   for c in t.scene.get_contacts():
    names=[b.entity.name for b in c.bodies]
    if any(n in potnames for n in names) and sum(np.linalg.norm(point.impulse) for point in c.points)>1e-8:
     for index,side in enumerate(['left','right']):contact[index]|=any(n in fingers[side] for n in names)
   contact_rows.append(contact);tcp_rows.append([float(np.linalg.norm(np.asarray(t.robot.get_left_tcp_pose()[:3])-np.asarray(t.pot.get_contact_point(0)[:3]))),float(np.linalg.norm(np.asarray(t.robot.get_right_tcp_pose()[:3])-np.asarray(t.pot.get_contact_point(1)[:3])))])
   for camera,item in obs['observation'].items():
    if 'rgb' in item:rgb.setdefault(camera,[]).append(np.asarray(item['rgb']).copy())
  def step():
   original();tick[0]+=1
   if tick[0]%15==0:capture()
  capture();t.scene.step=step;t.play_once();capture()
  legal,excess=t.planned_joints_legal();r.update(plan_success=bool(t.plan_success),official_success=bool(t.check_success()),joint_path_legal=bool(legal),joint_path_excess_rad=float(excess),frames=len(rows),physics_ticks=tick[0])
  # Unique timestamps; initial and final may share a sampling boundary.
  keep=[i for i in range(len(rows)) if i==0 or rows[i][0]!=rows[i-1][0]]
  drive=np.stack([rows[i][1] for i in keep]);r['accepted']=bool(r['plan_success'] and r['official_success'] and legal and np.isfinite(drive).all() and len(keep)>40 and {'left_camera','right_camera','head_camera'}.issubset(rgb))
  path=out/f"{'accepted' if r['accepted'] else 'rejected'}_seed{seed}.hdf5"
  with h5py.File(path,'w') as f:
   f.attrs.update(rgb_order='RGB',state_semantics='absolute drive targets; separate physical feedback, not substituted',instruction='Lift the pot with both arms.',model_id=r['model_id'],seed=seed,official_success=r['official_success'],quality_accepted=r['accepted'])
   f.create_dataset('sampled_finger_pot_contacts',data=np.asarray(contact_rows)[keep]);f.create_dataset('tcp_grasp_distance_m',data=np.asarray(tcp_rows)[keep]);f.create_dataset('physics_tick',data=[rows[i][0] for i in keep]);f.create_dataset('timestamp_s',data=[rows[i][0]*t.scene.get_timestep() for i in keep]);f.create_dataset('state_drive',data=drive);f.create_dataset('state_real_arm_legacy_command_gripper',data=np.stack([rows[i][2] for i in keep]));f.create_dataset('physical_finger_qpos_m',data=np.stack([rows[i][3] for i in keep]));f.create_dataset('pot_position',data=np.stack([rows[i][4] for i in keep]));f.create_dataset('official_success',data=[rows[i][5] for i in keep])
   for camera,frames in rgb.items():
    values=np.stack([frames[i] for i in keep]);assert values.dtype==np.uint8 and values.shape[-1]==3;f.create_dataset('rgb/'+camera,data=values,compression='gzip',compression_opts=1)
  r.update(file=str(path),bytes=path.stat().st_size,camera_shapes={k:list(v[0].shape) for k,v in rgb.items()},unique_frames=len(keep))
 except Exception:r['error']=traceback.format_exc()
 finally:
  r['wall_seconds']=time.perf_counter()-started;results.append(r);(out/'manifest.json').write_text(json.dumps(results,indent=2));print(json.dumps(r),flush=True)
  if hasattr(t,'scene'):t.close_env()
