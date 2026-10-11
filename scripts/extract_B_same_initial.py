"""Extract existing seed1 model predictions and original same-state expert targets."""
import json
from pathlib import Path
import h5py
import numpy as np
root=Path(__file__).resolve().parents[1]
p=np.load(root/'evaluation/lift_pot_gr00t_recovery_B_h5_d4_train_seed1_actions.npz')
z=np.load(root/'evaluation/offline_recovery_predictions_B.npz')
m=json.loads((root/'configs/dataset_split_manifest.json').read_text())
i=next(i for i,e in enumerate(m['splits']['train']) if e['source_episode']==1)
with h5py.File(root/'datasets/lift_pot/aloha-agilex_clean_50/data/episode1.hdf5') as f:q=f['joint_action/vector'][:]
assert np.max(np.abs(q[0]-p['policy_states'][0]))==0
assert np.max(np.abs(q[1:41]-z['targets'][i*5]))<1e-6
np.savez(root/'evaluation/B_live_seed1_same_initial_expert.npz',predictions=p['predictions'][0],expert=q[1:41],initial_state=q[0])
