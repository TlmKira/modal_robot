"""CPU-only saved data and AST audit. Never imports torch, SAPIEN or model code."""
import ast,json,hashlib,sys,importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
root=Path(__file__).resolve().parents[1];out=root/'results/C_cpu_audit_2026-10-11';out.mkdir(exist_ok=True)
paths=['scripts/BC_blind_runner.py','scripts/BC_blind_telemetry.py','scripts/gr00t_robotwin_interface.py','scripts/serve_gr00t_local.py','configs/aloha_gr00t_modality.py','scripts/recovery_control.py']
previous=json.loads((root/'results/end_of_day_2026-10-10/save_manifest.json').read_text());old={x['file']:x['sha256'] for x in previous['inventory']}
hashes={p:dict(sha256=hashlib.sha256((root/p).read_bytes()).hexdigest(),matches_saved_manifest=hashlib.sha256((root/p).read_bytes()).hexdigest()==old.get(p)) for p in paths}
cp={}
for m,name in [('B','lift_pot_rgb_B'),('C','lift_pot_C_v1')]:
 p=root/f'checkpoints/{name}/checkpoint-2000';idx=json.loads((p/'model.safetensors.index.json').read_text());cp[m]=dict(path=str(p),shards={n:(p/n).stat().st_size for n in set(idx['weight_map'].values())},metadata={n:(p/n).is_file() for n in ['config.json','statistics.json','processor_config.json','embodiment_id.json']})
spec=importlib.util.spec_from_file_location('cpu_interface',root/'scripts/gr00t_robotwin_interface.py');interface=importlib.util.module_from_spec(spec);spec.loader.exec_module(interface)
obs={'joint_action':{k:np.arange(n,dtype=float) for k,n in zip(interface.KEYS,[6,1,6,1])},'observation':{k:{'rgb':np.full((2,3,3),[v,2,3],dtype=np.uint8)} for k,v in zip(interface.CAMERAS.values(),[11,22,33])}}
converted=interface.observation_for_policy(obs,'test');assert [int(converted['video'][k][0,0,0,0,0]) for k in interface.CAMERAS]==[11,22,33]
acts={k:np.full((1,40,n),v,dtype=np.float32) for k,n,v in zip(interface.KEYS,[6,1,6,1],[.1,.2,.3,.4])};chunk=interface.action_chunk_for_robot(acts,np.tile([-10,10],(12,1)));assert np.allclose(chunk[0],[.1]*6+[.2]+[.3]*6+[.4])
# Extract only pure executor function; all simulation objects are CPU mocks.
tree=ast.parse((root/'scripts/BC_blind_runner.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute');space={'np':np};exec(compile(ast.Module(body=[fn],type_ignores=[]),'CPU-extracted-execute','exec'),space)
class Robot:
 def __init__(self):self.q=np.zeros(14);self.vel={}
 def get_left_arm_jointState(self):return self.q[:7].tolist()
 def get_right_arm_jointState(self):return self.q[7:].tolist()
 def set_arm_joints(self,q,v,s):self.q[0:6 if s=='left' else 0] if False else None;self.q[slice(0,6) if s=='left' else slice(7,13)]=q;self.vel[s]=np.asarray(v)
 def set_gripper(self,q,s):self.q[6 if s=='left' else 13]=q
robot=Robot();rec=SimpleNamespace(tick=0,contacts=lambda:{})
class Scene:
 def step(self):rec.tick+=1
scene=Scene();task=SimpleNamespace(robot=robot,scene=scene,take_action_cnt=0,check_success=lambda:False);space['execute'](task,chunk[0],rec,{'first_tick':None,'hold_ticks':[]});assert rec.tick==15 and np.allclose(robot.q,chunk[0]);assert np.allclose(robot.vel['left'],chunk[0,:6]/.06)
records=[];paired={};action_error=0;hold_corrections=[]
for m in ['B','C']:
 for p in sorted((root/'evaluation/BC_blind_v2').glob(m+'_seed*.json')):
  r=json.loads(p.read_text());z=np.load(p.with_name(p.stem+'_actions.npz'));pred=z['predictions'];executed=z['executed'];expected=np.concatenate([x[:5] for x in pred])[:len(executed)];assert expected.shape==executed.shape
  err=float(np.max(np.abs(expected-executed))) if executed.size else 0.;action_error=max(action_error,err)
  ts=r['telemetry'];initial=ts[0];paired[(m,r['seed'])]=r;first=min([v['physics_tick'] for v in r['first_contact'].values()],default=None)
  groups={'pre':[t for t in ts if first is None or t['physics_tick']<first],'post':[t for t in ts if first is not None and t['physics_tick']>=first]}
  row=dict(model=m,seed=r['seed'],official_success=r['official_success'],hold2s=r['hold_2s_continuous_official'],video_exists=Path(r['video']).is_file(),executed_matches_predicted_prefix_max=err)
  for side,sl in [('left',slice(0,6)),('right',slice(7,13))]:
   for stage,g in groups.items():row[f'{side}_{stage}_joint_mae_rad']=float(np.mean([np.mean(np.abs(np.asarray(t['drive'])[sl]-np.asarray(t['real_qpos_normalized_grippers'])[sl])) for t in g])) if g else None
   row[f'{side}_command_vs_drive_gripper_max']=max(abs(t[f'{side}_gripper_command']-t[f'{side}_gripper_drive_normalized']) for t in ts)
  if r['first_success_tick'] is not None:
   first=r['first_success_tick'];window=[h for h in r['hold_telemetry'] if first<h['tick']<=first+500];hold_corrections.append(dict(seed=r['seed'],logged_contact_s=r['hold_2s_both_contact_seconds'],corrected_contact_s=sum(h['both_contact'] for h in window)*.004,official_hold_same=(len(window)==500 and all(h['official'] for h in window))==r['hold_2s_continuous_official']))
  records.append(row)
seed_checks=[]
for seed in sorted(s for m,s in paired if m=='B'):
 b,c=paired[('B',seed)],paired[('C',seed)];a=np.asarray(b['telemetry'][0]['drive']);q=np.asarray(c['telemetry'][0]['drive']);realb=np.asarray(b['telemetry'][0]['real_qpos_normalized_grippers']);realc=np.asarray(c['telemetry'][0]['real_qpos_normalized_grippers'])
 seed_checks.append(dict(seed=seed,pot_initial_max_abs=float(np.max(np.abs(np.asarray(b['initial_pot_matrix'])-c['initial_pot_matrix']))),drive_initial_max_abs=float(np.max(np.abs(a-q))),real_initial_max_abs=float(np.max(np.abs(realb-realc)))))
summary=dict(checkpoints=cp,code_hashes=hashes,cpu_mock_checks='RGB camera sentinels/order and channel preservation, 14D action order, 15 ticks/60ms velocity and final target: all pass',all_executed_actions_match_prefix_max_abs=action_error,records=records,paired_initial_checks=seed_checks,hold_contact_window_corrections=hold_corrections,no_gpu_or_sapien_imported=not any(x=='torch' or x.startswith('sapien') for x in sys.modules))
(out/'evidence.json').write_text(json.dumps(summary,indent=2));print(json.dumps(dict(mock_checks='PASS',action_prefix_max_error=action_error,paired_initial_max=max(max(r[k] for k in ['pot_initial_max_abs','drive_initial_max_abs','real_initial_max_abs']) for r in seed_checks),gripper_command_drive_max=max(max(r['left_command_vs_drive_gripper_max'],r['right_command_vs_drive_gripper_max']) for r in records),hash_regressions=[p for p,h in hashes.items() if not h['matches_saved_manifest']],hold_official_unchanged=all(r['official_hold_same'] for r in hold_corrections),no_gpu_or_sapien_imported=summary['no_gpu_or_sapien_imported']),indent=2))
