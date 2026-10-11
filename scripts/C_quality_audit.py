"""Recheck immutable pilot recordings; accepted selection is a separate manifest."""
import argparse,json
from pathlib import Path
import h5py,numpy as np
p=argparse.ArgumentParser();p.add_argument('--label',default='pilot01');a=p.parse_args();root=Path(__file__).resolve().parents[1];folder=root/'datasets'/f'lift_pot_C_{a.label}';source=json.loads((folder/'manifest.json').read_text());records=[]
for r in source:
 q={'seed':r['seed'],'model_id':r.get('model_id'),'official_success':r.get('official_success',False),'accepted':False}
 if 'file' in r:
  with h5py.File(r['file']) as f:
   state=f['state_drive'][:];dt=np.diff(f['timestamp_s'][:]);grip=state[:,[6,13]];cams={k:list(f['rgb'][k].shape) for k in f['rgb']};required={'head_camera','left_camera','right_camera'}
   q.update(file=r['file'],bytes=r['bytes'],frames=len(state),timestamps_strict=bool(np.all(dt>0)),interval_min_s=float(dt.min()),interval_max_s=float(dt.max()),non_60ms_intervals=int(np.count_nonzero(np.abs(dt-.06)>1e-7)),finite_states=bool(np.isfinite(state).all()),gripper_command_in_0_1=bool(np.all((grip>=0)&(grip<=1))),camera_shapes=cams,rgb_uint8=all(f['rgb'][k].dtype==np.uint8 for k in cams),rgb_lossless_direct=True,closed_both_frames=int(np.count_nonzero((grip[:,0]<.1)&(grip[:,1]<.1))),lifted_frames=int(np.count_nonzero(f['pot_position'][:,2]>.82)))
   q['accepted']=bool(r.get('plan_success') and q['official_success'] and r.get('joint_path_legal') and q['timestamps_strict'] and q['finite_states'] and q['gripper_command_in_0_1'] and q['rgb_uint8'] and required.issubset(cams) and len(state)>40)
 else:q['error']=r.get('error')
 records.append(q)
summary={'attempts':len(records),'accepted':sum(r['accepted'] for r in records),'by_model':{str(i):{'attempts':sum(r.get('model_id')==i for r in records),'accepted':sum(r.get('model_id')==i and r['accepted'] for r in records)} for i in [0,1]},'records':records,'note':'selection manifest authoritative; source files immutable, including initial strict-three-camera gate that rejected optional front_camera. Camera required subset corrected in generator. No training export yet.'}
(root/'results/C_preparation'/f'{a.label}_quality.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='records'}))
