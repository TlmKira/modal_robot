"""Validate B's saved checkpoint headers and summarize measured training only."""
import csv
import hashlib
import json
from pathlib import Path
import struct

root = Path(__file__).resolve().parents[1]
checkpoint = Path('/tmp/gr00t_rgb_B_training/checkpoint-2000')
state = json.loads((checkpoint / 'trainer_state.json').read_text())
assert state['global_step'] == 2000
index = json.loads((checkpoint / 'model.safetensors.index.json').read_text())
headers = {}
shards = []
for filename in sorted(set(index['weight_map'].values())):
    path = checkpoint / filename
    with path.open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(size))
    for name, tensor in header.items():
        if name == '__metadata__':
            continue
        assert 0 <= tensor['data_offsets'][0] <= tensor['data_offsets'][1]
        assert tensor['data_offsets'][1] + 8 + size <= path.stat().st_size
        headers[name] = filename
    shards.append({'file': filename, 'bytes': path.stat().st_size})
assert headers == index['weight_map']
for filename in ['config.json', 'processor_config.json']:
    assert (checkpoint / filename).is_file()
loss = [row for row in state['log_history'] if 'loss' in row]
gpu = list(csv.DictReader((root / 'logs/lift-pot-rgb-B-gpu.csv').open()))
summary = {
    'checkpoint': str(checkpoint), 'global_step': state['global_step'],
    'first_logged_loss': loss[0]['loss'], 'last_logged_loss': loss[-1]['loss'],
    'checkpoint_tensors': len(headers), 'shards': shards,
    'peak_sampled_gpu_memory_mib': max(float(row['memory_used_mib']) for row in gpu),
    'config_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in checkpoint.glob('*.json')},
    'training_log': 'logs/lift-pot-rgb-B-training.log',
    'limitation': 'Training completed; loss and valid headers do not establish task success.',
}
(root / 'results/rgb_B_training_summary.json').write_text(json.dumps(summary, indent=2))
with (root / 'results/rgb_B_training_loss.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=['step', 'loss', 'grad_norm', 'learning_rate'])
    writer.writeheader()
    writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in loss)
print(json.dumps(summary), flush=True)
