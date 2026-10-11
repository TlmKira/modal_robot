"""Wait for the authorized training, summarize saved state, then verify inference."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--pid', type=int, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
while Path(f'/proc/{args.pid}').exists():
    time.sleep(10)
checkpoint = root / 'checkpoints/lift_pot_2000/checkpoint-2000'
state_path = checkpoint / 'trainer_state.json'
if not state_path.exists():
    raise RuntimeError('Training exited without checkpoint-2000 trainer state; inspect training log.')
state = json.loads(state_path.read_text())
assert state['global_step'] == 2000
loss = [item for item in state['log_history'] if 'loss' in item]
gpu = list(csv.DictReader((root / 'logs/lift-pot-2000-gpu.csv').open()))
summary = dict(steps=2000, checkpoint=str(checkpoint), first_logged_loss=loss[0]['loss'],
               last_logged_loss=loss[-1]['loss'],
               peak_sampled_gpu_memory_mib=max(float(row['memory_used_mib']) for row in gpu),
               limitation='Training loss only; closed-loop success remains unverified.')
(root / 'results/formal_training_summary.json').write_text(json.dumps(summary, indent=2))
with (root / 'results/formal_training_loss.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=['step', 'loss', 'grad_norm', 'learning_rate'])
    writer.writeheader()
    writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in loss)
print(json.dumps(summary), flush=True)
subprocess.run([str(root / '.venvs/gr00t/bin/python'), str(root / 'scripts/check_finetuned_policy.py'),
                '--checkpoint', str(checkpoint), '--output-prefix', 'formal_checkpoint'],
               cwd=root, check=True)
