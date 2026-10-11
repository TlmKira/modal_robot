"""Build isolated C v2.1 datasets from B drive data + fully qualified new experts. No training."""
import json,sys,subprocess,shutil
from pathlib import Path
import numpy as np,pandas as pd,h5py
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'simulation/RoboTwin'));from data.decode_image_bit import decode_image_bit
collected=json.loads((root/'evaluation/C_qualified_union.json').read_text());new=[r for r in collected['records'] if r['accepted']];assert len(new)>=50
manifest=json.loads((root/'configs/dataset_split_manifest.json').read_text());fps=50/3;pad=39;provenance=[];qc=[]
for split in ['train','validation','test']:
 src=root/f'datasets/lerobot/lift_pot_{split}_rgb_v23_terminal_pad';dst=root/f'datasets/lerobot/lift_pot_C_{split}_v1';assert not dst.exists();shutil.copytree(src/'meta',dst/'meta');info=json.loads((dst/'meta/info.json').read_text());tasks=[json.loads(l) for l in (dst/'meta/tasks.jsonl').read_text().splitlines()];instruction='Lift the pot with both arms.';taskidx=next((r['task_index'] for r in tasks if r['task']==instruction),None)
 if taskidx is None:taskidx=len(tasks);tasks.append({'task_index':taskidx,'task':instruction})
 entries=[('old',e) for e in manifest['splits'][split]]+([('new',e) for e in new] if split=='train' else []);episodes=[];globalindex=0
 for ei,(kind,e) in enumerate(entries):
  if kind=='old':
   df=pd.read_parquet(src/f"data/chunk-000/episode_{e['episode_index']:06d}.parquet");n=e['frames'];h=h5py.File(root/f"datasets/lift_pot/aloha-agilex_clean_50/data/episode{e['source_episode']}.hdf5");getimage=lambda cam,t:decode_image_bit(h[f'observation/{cam}/rgb'][min(t,n-1)]);seed=e['source_seed'];time_note='legacy original lacks exact timestamps; nominal 60ms, frame/order retained'
  else:
   h=h5py.File(e['file']);state=h['state_drive'][:];n=len(state)-1;assert np.all(np.diff(h['physics_tick'][:])==15);df=pd.DataFrame({'observation.state':[v.astype(np.float32) for v in state[:-1]],'action':[v.astype(np.float32) for v in state[1:]],'task_index':np.full(n,taskidx,dtype=np.int64)});tail=pd.concat([df.iloc[-1:]]*pad,ignore_index=True);tail['observation.state']=[state[-1].astype(np.float32)]*pad;tail['action']=[state[-1].astype(np.float32)]*pad;df=pd.concat([df,tail],ignore_index=True);getimage=lambda cam,t:h[f'rgb/{cam}'][min(t,n-1)];seed=e['seed'];time_note='exact 15x4ms fixed drive execution, includes measured 2s stable hold'
  size=len(df);assert size==n+pad;df['episode_index']=np.full(size,ei,dtype=np.int64);df['frame_index']=np.arange(size,dtype=np.int64);df['index']=np.arange(globalindex,globalindex+size,dtype=np.int64);df['timestamp']=np.arange(size,dtype=np.float32)*.06;globalindex+=size;target=dst/f'data/chunk-000/episode_{ei:06d}.parquet';target.parent.mkdir(parents=True,exist_ok=True);df.to_parquet(target,index=False)
  for cam,name in [('head_camera','cam_high'),('left_camera','cam_left_wrist'),('right_camera','cam_right_wrist')]:
   path=dst/f'videos/chunk-000/observation.images.{name}/episode_{ei:06d}.mp4';path.parent.mkdir(parents=True,exist_ok=True);proc=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','320x240','-r','50/3','-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','16','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE)
   first=np.asarray(getimage(cam,0));assert first.dtype==np.uint8 and first.shape==(240,320,3)
   for t in range(size):proc.stdin.write(np.ascontiguousarray(getimage(cam,t)).tobytes())
   proc.stdin.close();assert proc.wait()==0
   decoded=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-']);decoded=np.frombuffer(decoded,dtype=np.uint8).reshape(240,320,3);error=float(np.abs(decoded.astype(float)-first).mean());swapped=float(np.abs(decoded.astype(float)-first[...,::-1]).mean());assert error<5 and error<swapped;qc.append({'split':split,'episode':ei,'camera':cam,'RGB_MAE':error,'swapped_MAE':swapped})
  h.close();episodes.append({'episode_index':ei,'tasks':sorted({tasks[int(i)]['task'] for i in df.task_index.unique()}),'length':size});provenance.append({'split':split,'episode':ei,'seed':seed,'source':kind,'real_anchors':n,'stored_frames':size,'time_note':time_note});print(json.dumps(provenance[-1]),flush=True)
 info.update(fps=fps,total_episodes=len(entries),total_frames=globalindex,total_videos=len(entries)*3,total_tasks=len(tasks),splits={'train':f'0:{len(entries)}'})
 for feature in ['observation.state','action']:info['features'][feature]['names']=[[f'left_joint_{j}' for j in range(6)]+['left_gripper']+[f'right_joint_{j}' for j in range(6)]+['right_gripper']]
 for key in info['features']:
  if info['features'][key].get('dtype')=='video':info['features'][key]['info']['video.fps']=fps
 (dst/'meta/info.json').write_text(json.dumps(info,indent=2));(dst/'meta/tasks.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks));(dst/'meta/episodes.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in episodes))
 for name in ['stats.json','relative_stats.json']:(dst/'meta'/name).write_text('{}\n')
 (dst/'meta/C_provenance.json').write_text(json.dumps([e for e in provenance if e['split']==split],indent=2))
 (dst/'meta/terminal_padding.json').write_text(json.dumps([{'episode':e['episode'],'real_observation_frames':e['real_anchors'],'terminal_hold_targets':39,'stored_frames':e['stored_frames'],'synthetic_conditioning_anchors':0} for e in provenance if e['split']==split],indent=2))
 (dst/'meta/rgb_provenance.json').write_text(json.dumps({'source':'legacy raw JPEG official RGB decoder and qualified new lossless RGB HDF5','codec':'H264 CRF16 yuv420p','fps':fps,'state':'drive positions','new_time':'exact 60ms','old_time':'nominal 60ms; exact historical boundary timestamps unavailable'},indent=2))
(root/'evaluation/C_dataset_rgb_quality.json').write_text(json.dumps(qc,indent=2));(root/'evaluation/C_dataset_manifest.json').write_text(json.dumps(provenance,indent=2))
