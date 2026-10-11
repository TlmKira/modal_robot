"""Verify actual official shard index union, all real late states, and padded targets."""
import json,runpy
from pathlib import Path
import numpy as np,pandas as pd
from gr00t.data.dataset.sharded_single_step_dataset import ShardedSingleStepDataset,extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
root=Path(__file__).resolve().parents[1];cfg=runpy.run_path(str(root/'configs/aloha_gr00t_modality.py'))['config'];manifest=json.loads((root/'configs/dataset_split_manifest.json').read_text());rows=[]
for split in ['train','validation','test']:
 path=root/f'datasets/lerobot/lift_pot_{split}_rgb_v23_terminal_pad';old=root/f'datasets/lerobot/lift_pot_{split}_v21';ds=ShardedSingleStepDataset(path,EmbodimentTag.NEW_EMBODIMENT,cfg,shard_size=256);selected={i:[] for i in range(len(manifest['splits'][split]))}
 for shard in ds.sharded_episodes:
  for ei,indices in shard:selected[int(ei)].extend(indices.tolist())
 for e in manifest['splits'][split]:
  ei=e['episode_index'];n=e['frames'];assert sorted(selected[ei])==list(range(n));fn=f'data/chunk-000/episode_{ei:06d}.parquet';data=pd.read_parquet(path/fn);orig=pd.read_parquet(old/fn)
  for column in ['observation.state','action']:assert np.array_equal(np.stack(data[column].iloc[:n]),np.stack(orig[column]))
  state=np.stack(data['observation.state'].iloc[:n]);closed=int(((state[:,6]<.1)&(state[:,13]<.1)).sum());episode=ds.episode_loader[ei];point=extract_step_data(episode,n-1,cfg,EmbodimentTag.NEW_EMBODIMENT,False)
  assert all(np.asarray(point.actions[k]).shape==(40,6 if 'arm' in k else 1) for k in cfg['action'].modality_keys)
  assert all(np.allclose(point.actions[k],point.actions[k][0]) for k in cfg['action'].modality_keys)
  rows.append({'split':split,'episode':ei,'real_anchors':n,'synthetic_anchors_selected':0,'closed_gripper_anchors':closed,'last_real_observation_horizon40_valid':True,'original_states_actions_equal':True});print(rows[-1],flush=True)
result={'rows':rows,'train_real_anchors':sum(r['real_anchors'] for r in rows if r['split']=='train'),'train_closed_gripper_anchors':sum(r['closed_gripper_anchors'] for r in rows if r['split']=='train'),'synthetic_conditioning_anchors':0,'scope':'terminal holds are future targets only; never a simulation result'};assert result['train_real_anchors']==4444 and result['train_closed_gripper_anchors']==990
(root/'evaluation/terminal_pad_sampler_check.json').write_text(json.dumps(result,indent=2));print(result['train_real_anchors'],result['train_closed_gripper_anchors'])
