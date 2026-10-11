"""Check actual B v23 videos against official decoded source RGB, without inference."""
import json,sys
from pathlib import Path
import cv2,h5py,numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'simulation/RoboTwin'))
from data.decode_image_bit import decode_image_bit
manifest=json.loads((root/'configs/dataset_split_manifest.json').read_text());samples=[]
for split,entries in manifest['splits'].items():
 for e in entries:
  with h5py.File(root/f"datasets/lift_pot/aloha-agilex_clean_50/data/episode{e['source_episode']}.hdf5") as f:
   for key,cam in [('cam_high','head_camera'),('cam_left_wrist','left_camera'),('cam_right_wrist','right_camera')]:
    videos=list((root/f'datasets/lerobot/lift_pot_{split}_rgb_v23_terminal_pad/videos').rglob(f"episode_{e['episode_index']:06d}.mp4"));path=next(v for v in videos if key in str(v))
    capture=cv2.VideoCapture(str(path));ok,bgr=capture.read();capture.release();assert ok
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);source=decode_image_bit(f[f'observation/{cam}/rgb'][0]);assert rgb.shape==source.shape and rgb.dtype==np.uint8
    samples.append({'split':split,'source_episode':e['source_episode'],'camera':cam,'shape':list(rgb.shape),'rgb_mae':float(np.abs(rgb.astype(float)-source).mean()),'swapped_mae':float(np.abs(rgb.astype(float)-source[...,::-1]).mean())})
result={'dataset':'rgb_v23_terminal_pad','sample_frames':len(samples),'rgb_better_count':sum(s['rgb_mae']<s['swapped_mae'] for s in samples),'mean_rgb_mae':float(np.mean([s['rgb_mae'] for s in samples])),'mean_swapped_mae':float(np.mean([s['swapped_mae'] for s in samples])),'samples':samples,'scope':'First frame of all 50 episodes, all 3 cameras; not every frame.'}
assert result['rgb_better_count']==result['sample_frames']
(root/'evaluation/B_rgb_pixel_audit.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='samples'})
