import numpy as np
import sapien,imageio.v2 as imageio
from pathlib import Path
from recovery_fk import poses
root=Path(__file__).resolve().parents[1]
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
 r.update(state_semantics='drive command to policy; measured qpos for diagnostics only',drive_command=drive.tolist(),measured_entity_qpos={side:getattr(t.robot,side+'_entity').get_qpos().tolist() for side in ['left','right']},measured_qvel={side:getattr(t.robot,side+'_entity').get_qvel().tolist() for side in ['left','right']},pot_pose=pose.to_transformation_matrix().tolist(),actual_tcp_pose=[np.asarray(t.robot.get_left_tcp_pose()).tolist(),np.asarray(t.robot.get_right_tcp_pose()).tolist()])
 for i,s,offset in [(0,'left',0),(1,'right',7)]:
  contactpoint=np.asarray(t.pot.get_contact_point(i)[:3]);r.update({f'{s}_grasp_distance_m':float(np.linalg.norm(actual[i]-contactpoint)),f'{s}_drive_tcp_grasp_distance_m':float(np.linalg.norm(pd[i][:3,3]-contactpoint)),f'{s}_tcp_tracking_m':float(np.linalg.norm(pd[i][:3,3]-actual[i])),f'{s}_joint_mae_rad':float(np.abs(drive[offset:offset+6]-real[offset:offset+6]).mean()),f'{s}_joint_max_error_rad':float(np.abs(drive[offset:offset+6]-real[offset:offset+6]).max()),f'{s}_gripper_command':float(drive[offset+6]),f'{s}_gripper_drive_normalized':float((getattr(t.robot,s+'_gripper')[0][0].get_drive_target()[0]-getattr(t.robot,s+'_gripper_scale')[0])/(getattr(t.robot,s+'_gripper_scale')[1]-getattr(t.robot,s+'_gripper_scale')[0])),f'{s}_gripper_actual_normalized':float(grip[s][0]),f'{s}_finger_actual_normalized':grip[s],f'{s}_finger_pot_contact':contact.get(s,False),f'{s}_arm_pot_contact':contact.get(s+'_arm',False),f'{s}_contact_impulse_ns':contact.get(s+'_impulse',0.0)})
 r['official_components']={'height_above_082':r['pot_height_m']>.82,'left_within_003':r['left_grasp_distance_m']<.03,'right_within_003':r['right_grasp_distance_m']<.03,'upright_above_08':r['pot_up_dot']>.8}
 return r

class Recorder:
 def __init__(self,t,seed,variant,run_label='v2'):
  self.t=t;self.tick=0;self.rows=[];self.events=[];self.first={};self.contact_ticks={'left':0,'right':0,'both':0};self.video=root/f'videos/BC_blind_{run_label}_{variant}_seed{seed}_raw.mp4';assert not self.video.exists();self.writer=imageio.get_writer(self.video,fps=250/15,codec='libx264',quality=8,macro_block_size=1)
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
 def record_event(self,event,official):
  row=snapshot(self.t,self.tick,self.contacts());row.update(event=event,event_official=bool(official));self.events.append(row)
 def close(self):
  self.t.scene.step=self.original
  self.rows.append(snapshot(self.t,self.tick,self.contacts()));self.capture();self.writer.close()

