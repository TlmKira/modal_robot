"""Read-only analysis of the completed B test20; never launches simulation/training."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from recovery_fk import poses
root=Path(__file__).resolve().parents[1]
rs=json.loads((root/'evaluation/gr00t_lift_pot_recovery_B_h5_d4_test20.json').read_text())
assert len(rs)==20 and len({r['seed'] for r in rs})==20
rows=[];track=[];interfaces=[];fkchecks=[]
for r in rs:
 calls=r['inference_calls'];last=calls[-1]
 p=np.load(root/f"evaluation/lift_pot_gr00t_recovery_B_h5_d4_test20_seed{r['seed']}_actions.npz")
 sent=p['predictions'][:,:5,:].reshape(-1,14)[:len(p['executed'])]
 interfaces.append(float(np.max(np.abs(sent-p['executed']))))
 for i,c in enumerate(calls):
  commanded=p['states'][i];real=np.asarray(c['real_joints']);pc,pr=poses(commanded),poses(real)
  track.append({'seed':r['seed'],'action_count':c['action_count'],'joint_mae_rad':float(np.abs(commanded-real)[[0,1,2,3,4,5,7,8,9,10,11,12]].mean()),'left_tcp_tracking_m':float(np.linalg.norm(pc[0][:3,3]-pr[0][:3,3])),'right_tcp_tracking_m':float(np.linalg.norm(pc[1][:3,3]-pr[1][:3,3]))})
 row={'seed':r['seed'],'success':r['success'],'actions':r['actions'],'model_id':r['model_id'],'last_observed_action_count':last['action_count'],'last_observed_left_m':last['left_grasp_distance'],'last_observed_right_m':last['right_grasp_distance'],'min_observed_left_m':min(c['left_grasp_distance'] for c in calls),'min_observed_right_m':min(c['right_grasp_distance'] for c in calls),'best_simultaneous_max_m':min(max(c['left_grasp_distance'],c['right_grasp_distance']) for c in calls),'final_left_m':None,'final_right_m':None,'final_height_m':r['final_pot_position'][2],'final_pot_up_dot':None,'error':r.get('error',''),'raw_video':r['video']}
 if 'final_left_tcp' in r:
  for side in ['left','right']:row[f'final_{side}_m']=float(np.linalg.norm(np.asarray(r[f'final_{side}_tcp'])-r[f'final_{side}_grasp']))
  q=r['final_pot_quaternion'];row['final_pot_up_dot']=float(Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()[2,2])
 rows.append(row)
with (root/'results/B_test20_grasp_distances.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (root/'results/B_test20_joint_tracking.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(track[0]));w.writeheader();w.writerows(track)
def stats(values):
 a=np.asarray(values,dtype=float);return {'n':len(a),'mean':float(a.mean()),'min':float(a.min()),'max':float(a.max())}
summary={'successes':sum(r['success'] for r in rs),'failures':sum(not r['success'] for r in rs),'last_observed_distance_all20':{s:stats([r[f'last_observed_{s}_m'] for r in rows]) for s in ['left','right']},'final_distance_available':{s:stats([r[f'final_{s}_m'] for r in rows if r[f'final_{s}_m'] is not None]) for s in ['left','right']},'minimum_observed_distance_all20':{s:stats([r[f'min_observed_{s}_m'] for r in rows]) for s in ['left','right']},'best_simultaneous_max_distance':stats([r['best_simultaneous_max_m'] for r in rows]),'model_target_to_executed_drive_max_abs':max(interfaces),'tracking':{key:stats([r[key] for r in track]) for key in ['joint_mae_rad','left_tcp_tracking_m','right_tcp_tracking_m']},'terminal_missing_seeds':[r['seed'] for r in rows if r['final_left_m'] is None]}
# Same observation/input state: B's saved initial offline chunk vs expert dataset actions.
p=np.load(root/'evaluation/offline_recovery_predictions_B.npz');pred=p['predictions'][::5];target=p['targets'][::5]
initial=[]
for split,lo,hi in [('train',0,40),('validation',40,45)]:
 for h in [1,5,10,20,40]:
  x,y=pred[lo:hi,h-1],target[lo:hi,h-1];e=np.abs(x-y);rec={'split':split,'horizon':h,'samples':hi-lo,'left_joint_mae_rad':float(e[:,:6].mean()),'right_joint_mae_rad':float(e[:,7:13].mean()),'left_gripper_mae':float(e[:,6].mean()),'right_gripper_mae':float(e[:,13].mean())}
  d=[(poses(a),poses(b)) for a,b in zip(x,y)]
  for i,s in enumerate(['left','right']):rec[f'{s}_tcp_error_m']=float(np.mean([np.linalg.norm(a[i][:3,3]-b[i][:3,3]) for a,b in d]))
  initial.append(rec)
summary['same_initial_observation_expert_comparison']=initial
summary['same_initial_comparison_scope']='All 40 training and 5 validation episode initial RGB/proprioception observations. Existing saved predictions, no new simulation. No expert trajectories available for unseen seeds 200000..200019; no matched unseen expert claim.'
summary['phase_tracking']={}
for lo,hi in [(0,50),(50,100),(100,150),(150,201)]:
 subset=[r for r in track if lo<=r['action_count']<hi]
 summary['phase_tracking'][f'{lo}-{hi-1}']={key:stats([r[key] for r in subset]) for key in ['joint_mae_rad','left_tcp_tracking_m','right_tcp_tracking_m']}
scene=json.loads((root/'datasets/lift_pot/aloha-agilex_clean_50/scene_info.json').read_text())
manifest=json.loads((root/'configs/dataset_split_manifest.json').read_text())
summary['demonstration_model_coverage']={split:{f'base{k}':sum(scene[f"episode_{e['source_episode']}"]['info']['{A}'].endswith(f'base{k}') for e in entries) for k in [0,1]} for split,entries in manifest['splits'].items()}
summary['independent_model_results']={f'base{k}':{'episodes':sum(r['model_id']==k for r in rs),'successes':sum(r['model_id']==k and r['success'] for r in rs)} for k in [0,1]}
if (root/'evaluation/B_live_seed1_same_initial_expert.npz').exists():
 z=np.load(root/'evaluation/B_live_seed1_same_initial_expert.npz');live=[]
 for h in [1,5,10,20,40]:
  a,b=z['predictions'][h-1],z['expert'][h-1];e=np.abs(a-b);pa,pb=poses(a),poses(b)
  live.append({'horizon':h,'left_joint_mae_rad':float(e[:6].mean()),'right_joint_mae_rad':float(e[7:13].mean()),'left_gripper_mae':float(e[6]),'right_gripper_mae':float(e[13]),'left_tcp_error_m':float(np.linalg.norm(pa[0][:3,3]-pb[0][:3,3])),'right_tcp_error_m':float(np.linalg.norm(pa[1][:3,3]-pb[1][:3,3]))})
 summary['live_seed1_same_initial_expert_comparison']=live
 summary['live_initial_proprioception_max_abs_difference']=float(np.max(np.abs(z['initial_state']-np.load(root/'evaluation/lift_pot_gr00t_recovery_B_h5_d4_train_seed1_actions.npz')['policy_states'][0])))
(root/'evaluation/B_test20_diagnostic_summary.json').write_text(json.dumps(summary,indent=2))
fig,axes=plt.subplots(1,3,figsize=(14,4))
for r in rs:
 c=r['inference_calls'];x=[v['action_count']*.06 for v in c]
 for ax,key in zip(axes,['left_grasp_distance','right_grasp_distance','pot_height']):ax.plot(x,[v[key] for v in c],alpha=.5,lw=1)
for ax,side in zip(axes[:2],['Left','Right']):ax.axhline(.03,color='red',ls='--',label='Official 3 cm');ax.set(title=f'{side} TCP to moving pot contact',ylabel='Distance (m)');ax.legend(fontsize=8)
axes[2].axhline(.82,color='red',ls='--');axes[2].set(title='Pot height',ylabel='Height (m)')
for ax in axes:ax.set_xlabel('Executed simulated time (s)');ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(root/'results/B_test20_contact_error_curves.png',dpi=160)
fig,axes=plt.subplots(1,2,figsize=(10,4))
for split in ['train','validation']:
 rr=[r for r in initial if r['split']==split]
 for side in ['left','right']:
  axes[0].plot([r['horizon'] for r in rr],[r[f'{side}_joint_mae_rad'] for r in rr],marker='o',label=f'{split} {side}')
  axes[1].plot([r['horizon'] for r in rr],[r[f'{side}_tcp_error_m']*100 for r in rr],marker='o',label=f'{split} {side}')
for ax,y in zip(axes,['Joint MAE (rad)','TCP error vs expert (cm)']):ax.set(xlabel='Predicted step',ylabel=y);ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(root/'results/B_same_initial_expert_error_curves.png',dpi=160)
fig,axes=plt.subplots(1,3,figsize=(14,4))
for ax,seed in zip(axes,[200004,200007,200013]):
 r=next(r for r in rs if r['seed']==seed);c=r['inference_calls'];t=[v['action_count']*.06 for v in c]
 ax.plot(t,[v['left_grasp_distance']*100 for v in c],label='Left contact error');ax.plot(t,[v['right_grasp_distance']*100 for v in c],label='Right contact error')
 tr=[v for v in track if v['seed']==seed]
 ax.plot(t,[v['left_tcp_tracking_m']*100 for v in tr],ls=':',label='Left drive-real TCP');ax.plot(t,[v['right_tcp_tracking_m']*100 for v in tr],ls=':',label='Right drive-real TCP')
 ax.axhline(3,color='red',ls='--');ax.set(title=f'Seed {seed}',xlabel='Simulated time (s)',ylabel='Distance (cm)');ax.legend(fontsize=6);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(root/'results/B_representative_tracking_curves.png',dpi=160)
print(json.dumps(summary,indent=2))
