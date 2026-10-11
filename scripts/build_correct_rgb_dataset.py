"""Create new v2.1 datasets directly from official JPEG RGB; no old data changes."""
import json,os,sys,subprocess,shutil,hashlib
from pathlib import Path
import h5py,numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'simulation/RoboTwin'))
from data.decode_image_bit import decode_image_bit
manifest=json.loads((root/'configs/dataset_split_manifest.json').read_text());output=[]
for split,entries in manifest['splits'].items():
 src=root/f'datasets/lerobot/lift_pot_{split}_v21'; dst=root/f'datasets/lerobot/lift_pot_{split}_rgb_v22'
 if dst.exists():raise FileExistsError(f'Preserving existing new dataset {dst}')
 dst.mkdir();shutil.copytree(src/'meta',dst/'meta');shutil.copytree(src/'data',dst/'data',copy_function=os.link)
 info=json.loads((dst/'meta/info.json').read_text())
 for value in info['features'].values():
  if value.get('dtype')=='video':value['info']['video.codec']='h264'
 (dst/'meta/info.json').write_text(json.dumps(info,indent=2))
 for e in entries:
  with h5py.File(root/f"datasets/lift_pot/aloha-agilex_clean_50/data/episode{e['source_episode']}.hdf5") as f:
   for cam,name in [('head_camera','cam_high'),('left_camera','cam_left_wrist'),('right_camera','cam_right_wrist')]:
    video=dst/f"videos/chunk-000/observation.images.{name}/episode_{e['episode_index']:06d}.mp4";video.parent.mkdir(parents=True,exist_ok=True)
    arrays=f[f'observation/{cam}/rgb']; first=decode_image_bit(arrays[0]);height,width=first.shape[:2]
    proc=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','15','-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','16','-pix_fmt','yuv420p',str(video)],stdin=subprocess.PIPE)
    for t in range(e['frames']):proc.stdin.write(np.ascontiguousarray(decode_image_bit(arrays[t])).tobytes())
    proc.stdin.close();assert proc.wait()==0
  output.append({'split':split,**e});print(json.dumps(output[-1]),flush=True)
 (dst/'meta/rgb_provenance.json').write_text(json.dumps({'source':'original RoboTwin HDF5 JPEG decoded with official marker-aware decode_image_bit; no extra channel conversion','codec':'libx264 CRF16 yuv420p 15Hz','source_split_manifest':'configs/dataset_split_manifest.json','frames':'image[t], state=q[t], action=q[t+1]; original split/parquets/statistics retained','parquet_storage':'hardlinks to immutable originals; no edits permitted'},indent=2))
(root/'evaluation/correct_rgb_dataset_manifest.json').write_text(json.dumps(output,indent=2))
