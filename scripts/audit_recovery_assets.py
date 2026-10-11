"""Read-only checkpoint headers, environment metadata and JPEG/video channel audit."""
import json,struct,subprocess,sys,os,hashlib,argparse
from pathlib import Path
import cv2,h5py,numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--dataset-suffix',default='v21',choices=['v21','rgb_v22']);args=parser.parse_args()
root=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(root/'simulation/RoboTwin'))
from data.decode_image_bit import decode_image_bit,_has_rgb_marker
cp=root/'checkpoints/lift_pot_2000/checkpoint-2000'; index=json.loads((cp/'model.safetensors.index.json').read_text()); headers={}; files=[]
for name in sorted(set(index['weight_map'].values())):
    path=cp/name
    with path.open('rb') as f: size=struct.unpack('<Q',f.read(8))[0]; header=json.loads(f.read(size))
    for key,value in header.items():
        if key!='__metadata__': assert value['data_offsets'][1]+8+size<=path.stat().st_size; headers[key]=name
    files.append({'file':name,'size_bytes':path.stat().st_size,'header_sha256':hashlib.sha256(json.dumps(header,sort_keys=True).encode()).hexdigest()})
assert headers==index['weight_map']; result={'checkpoint_tensors':len(headers),'shards':files,'config_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cp.glob('*.json')},'storage':subprocess.check_output(['df','-hT','/','/workspace'],text=True),'quota_tool_available':bool(subprocess.run(['bash','-c','command -v quota'],capture_output=True).stdout),'rgb_samples':[]}
manifest=json.loads((root/'configs/dataset_split_manifest.json').read_text())
for split,entries in manifest['splits'].items():
    for e in entries:
        with h5py.File(root/f"datasets/lift_pot/aloha-agilex_clean_50/data/episode{e['source_episode']}.hdf5") as f:
            for key,cam in [('head','head_camera'),('left_wrist','left_camera'),('right_wrist','right_camera')]:
                source=decode_image_bit(f[f'observation/{cam}/rgb'][0]); marked=_has_rgb_marker(f[f'observation/{cam}/rgb'][0]); video=root/f"datasets/lerobot/lift_pot_{split}_{args.dataset_suffix}/videos/chunk-000/observation.images.{key}/episode_{e['episode_index']:06d}.mp4"
                if not video.exists():
                    matches=list((root/f'datasets/lerobot/lift_pot_{split}_{args.dataset_suffix}/videos').rglob(f"episode_{e['episode_index']:06d}.mp4")); video=next(v for v in matches if {'head':'cam_high','left_wrist':'cam_left_wrist','right_wrist':'cam_right_wrist'}[key] in str(v))
                bits=subprocess.check_output(['ffmpeg','-v','error',*(['-c:v','libdav1d'] if args.dataset_suffix=='v21' else []),'-i',str(video),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-']); frame=np.frombuffer(bits,dtype=np.uint8).reshape(source.shape)
                result['rgb_samples'].append({'split':split,'source_episode':e['source_episode'],'camera':key,'marked':marked,'shape':list(source.shape),'dtype':str(source.dtype),'range':[int(source.min()),int(source.max())],'rgb_mae':float(np.abs(frame.astype(float)-source).mean()),'swapped_mae':float(np.abs(frame.astype(float)-source[...,::-1]).mean())})
(root/('evaluation/recovery_asset_audit.json' if args.dataset_suffix=='v21' else 'evaluation/correct_rgb_pixel_audit.json')).write_text(json.dumps(result,indent=2)); print('checkpoint',len(headers),'RGB pairs',len(result['rgb_samples'])); print(result['storage'])
