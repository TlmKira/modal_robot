"""Official loader check for new RGB versions; no training or statistics writes."""
import json,runpy,hashlib
from pathlib import Path
import numpy as np
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
root=Path(__file__).resolve().parents[1];cfg=runpy.run_path(str(root/'configs/aloha_gr00t_modality.py'))['config'];manifest=json.loads((root/'configs/dataset_split_manifest.json').read_text());rows=[]
for split in ['train','validation','test']:
 new=root/f'datasets/lerobot/lift_pot_{split}_rgb_v22';old=root/f'datasets/lerobot/lift_pot_{split}_v21';loader=LeRobotEpisodeLoader(new,cfg)
 for e in manifest['splits'][split]:
  ep=loader[e['episode_index']];assert len(ep)==e['frames']; assert all(np.isfinite(np.stack(ep[f'action.{k}'].to_list())).all() for k in cfg['action'].modality_keys)
  for k in cfg['video'].modality_keys: assert np.asarray(ep[f'video.{k}'].iloc[0]).shape==(240,320,3)
  fn=f"data/chunk-000/episode_{e['episode_index']:06d}.parquet";assert hashlib.sha256((new/fn).read_bytes()).digest()==hashlib.sha256((old/fn).read_bytes()).digest()
  rows.append({'split':split,'episode':e['episode_index'],'frames':len(ep),'parquet_equal':True,'loader_ok':True});print(rows[-1],flush=True)
 for fn in ['stats.json','relative_stats.json']:assert (new/'meta'/fn).read_bytes()==(old/'meta'/fn).read_bytes()
(root/'evaluation/correct_rgb_loader_check.json').write_text(json.dumps(rows,indent=2))
