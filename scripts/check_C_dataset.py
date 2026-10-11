import json,runpy,hashlib
from pathlib import Path
import pandas as pd,numpy as np
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset,extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
root=Path(__file__).resolve().parents[1];cfg=runpy.run_path(str(root/'configs/aloha_gr00t_modality.py'))['config'];result=[]
for split in ['train','validation','test']:
 path=root/f'datasets/lerobot/lift_pot_C_{split}_v1';prov=json.loads((path/'meta/C_provenance.json').read_text());ds=ShardedSingleStepDataset(path,EmbodimentTag.NEW_EMBODIMENT,cfg,shard_size=256);selected={i:[] for i in range(len(prov))}
 for shard in ds.sharded_episodes:
  for i,indices in shard:selected[int(i)].extend(indices.tolist())
 for e in prov:
  i=e['episode'];n=e['real_anchors'];assert sorted(selected[i])==list(range(n));df=pd.read_parquet(path/f'data/chunk-000/episode_{i:06d}.parquet');state=np.stack(df['observation.state']);action=np.stack(df.action);assert np.all(np.isfinite(state)) and np.all(np.isfinite(action));assert np.allclose(np.diff(df.timestamp),.06,atol=2e-6);assert np.all((action[:,[6,13]]>=0)&(action[:,[6,13]]<=1));ep=ds.episode_loader[i]
  for anchor in [0,n//2,n-1]:
   point=extract_step_data(ep,anchor,cfg,EmbodimentTag.NEW_EMBODIMENT,False);assert all(np.asarray(point.actions[k]).shape==(40,6 if 'arm' in k else 1) for k in cfg['action'].modality_keys)
  result.append({**e,'sampler_real_anchor_count':len(selected[i]),'synthetic_conditioning_anchors':0,'closed_both_anchors':int(np.count_nonzero((state[:n,6]<.1)&(state[:n,13]<.1)))})
 for name in ['stats.json','relative_stats.json']:
  assert (path/'meta'/name).read_bytes()==(root/'datasets/lerobot/lift_pot_C_train_v1/meta'/name).read_bytes()
(root/'evaluation/C_dataset_loader_check.json').write_text(json.dumps({'records':result,'statistics_source':'train only; copied byte-identically to validation/test','stats_sha256':{name:hashlib.sha256((root/'datasets/lerobot/lift_pot_C_train_v1/meta'/name).read_bytes()).hexdigest() for name in ['stats.json','relative_stats.json']}},indent=2));print('C loader/checks passed',len(result))
