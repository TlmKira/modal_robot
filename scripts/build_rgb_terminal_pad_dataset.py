"""Training-only terminal target padding restores all REAL observation anchors.

39 appended hold targets for a 40-step horizon; the existing official sampler
then conditions only on original real observations, never on synthetic holds.
"""
import os,sys,json,shutil,subprocess
from pathlib import Path
import numpy as np,pandas as pd,h5py
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'simulation/RoboTwin'))
from data.decode_image_bit import decode_image_bit
manifest=json.loads((root/'configs/dataset_split_manifest.json').read_text());pad=39;records=[]
for split,entries in manifest['splits'].items():
 src=root/f'datasets/lerobot/lift_pot_{split}_rgb_v22';dst=root/f'datasets/lerobot/lift_pot_{split}_rgb_v23_terminal_pad'
 if dst.exists():raise FileExistsError(f'Preserving {dst}')
 shutil.copytree(src/'meta',dst/'meta')
 for stats_name in ['stats.json','relative_stats.json']:
  shutil.copy2(dst/'meta'/stats_name,dst/'meta'/stats_name.replace('.json','_pre_refresh.json'))
  (dst/'meta'/stats_name).write_text('{}\n')
 episodes=[json.loads(l) for l in (dst/'meta/episodes.jsonl').read_text().splitlines()];global_index=0
 for e in entries:
  fn=f"data/chunk-000/episode_{e['episode_index']:06d}.parquet";df=pd.read_parquet(src/fn);n=len(df);assert n==e['frames'];tail=pd.concat([df.iloc[-1:]]*pad,ignore_index=True);final_action=np.asarray(df['action'].iloc[-1]).copy()
  tail['observation.state']=[final_action.copy() for _ in range(pad)];tail['action']=[final_action.copy() for _ in range(pad)];combined=pd.concat([df,tail],ignore_index=True);combined['frame_index']=np.arange(n+pad);combined['timestamp']=np.arange(n+pad,dtype=np.float32)/15;combined['index']=np.arange(global_index,global_index+n+pad);global_index+=n+pad
  out=dst/fn;out.parent.mkdir(parents=True,exist_ok=True);combined.to_parquet(out,index=False);episodes[e['episode_index']]['length']=n+pad
  # Re-encode once directly from original RGB JPEGs. Held images are never
  # selected as conditioning anchors by this dataset/sampler combination.
  with h5py.File(root/f"datasets/lift_pot/aloha-agilex_clean_50/data/episode{e['source_episode']}.hdf5") as f:
   for cam,name in [('head_camera','cam_high'),('left_camera','cam_left_wrist'),('right_camera','cam_right_wrist')]:
    encoded=f[f'observation/{cam}/rgb'];first=decode_image_bit(encoded[0]);h,w=first.shape[:2];video=dst/f"videos/chunk-000/observation.images.{name}/episode_{e['episode_index']:06d}.mp4";video.parent.mkdir(parents=True,exist_ok=True)
    proc=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','15','-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','16','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
    for t in range(n):proc.stdin.write(np.ascontiguousarray(decode_image_bit(encoded[t])).tobytes())
    final_rgb=decode_image_bit(encoded[min(n,len(encoded)-1)]).tobytes()
    for _ in range(pad):proc.stdin.write(final_rgb)
    proc.stdin.close();assert proc.wait()==0
  records.append({'split':split,**e,'real_observation_frames':n,'terminal_hold_targets':pad,'stored_frames':n+pad,'official_sampler_anchor_count':n,'terminal_padding_is_training_only':True});print(json.dumps(records[-1]),flush=True)
 info=json.loads((dst/'meta/info.json').read_text());info['total_frames']=global_index;(dst/'meta/info.json').write_text(json.dumps(info,indent=2));(dst/'meta/episodes.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in episodes));(dst/'meta/terminal_padding.json').write_text(json.dumps([r for r in records if r['split']==split],indent=2))
(root/'evaluation/rgb_terminal_pad_manifest.json').write_text(json.dumps(records,indent=2))
