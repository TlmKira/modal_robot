"""Local-only bridge to the real checkpoint in its existing Python environment."""
import argparse
import json
from multiprocessing.connection import Listener
from pathlib import Path
import runpy
import time
import traceback
import numpy as np
import torch
from gr00t.policy.gr00t_policy import Gr00tPolicy

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='checkpoints/lift_pot_2000/checkpoint-2000')
parser.add_argument('--port', type=int, default=6007)
parser.add_argument('--denoising-steps', type=int, default=4)
parser.add_argument('--ensemble', type=int, default=1)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
runpy.run_path(str(root / 'configs/aloha_gr00t_modality.py'))
checkpoint = (root / args.checkpoint).resolve()
policy = Gr00tPolicy(embodiment_tag='NEW_EMBODIMENT', model_path=str(checkpoint), device='cuda:0')
policy.model.action_head.num_inference_timesteps = args.denoising_steps
listener = Listener(('127.0.0.1', args.port), authkey=b'local-gr00t-simulation')
print(json.dumps({'ready': True, 'checkpoint': str(checkpoint), 'port': args.port,
                  'denoising_steps':policy.model.action_head.num_inference_timesteps, 'ensemble':args.ensemble}), flush=True)
stop = False
while not stop:
    conn = listener.accept()
    try:
        while True:
            request = conn.recv()
            if request['op'] == 'shutdown':
                stop = True
                conn.send({'ok': True})
                break
            if request['op'] == 'reset':
                policy.reset()
                torch.manual_seed(request['seed'])
                conn.send({'ok': True})
                continue
            steps = int(request.get('denoising_steps', args.denoising_steps))
            ensemble = int(request.get('ensemble', args.ensemble))
            if not 1 <= steps <= 64 or not 1 <= ensemble <= 16:
                raise ValueError('Invalid inference sampling configuration')
            policy.model.action_head.num_inference_timesteps = steps
            start = time.perf_counter()
            samples = [policy.get_action(request['observation'])[0] for _ in range(ensemble)]
            actions = {key: np.mean([sample[key] for sample in samples],axis=0) for key in samples[0]}
            torch.cuda.synchronize()
            conn.send({'actions': {k: np.asarray(v) for k,v in actions.items()},
                       'inference_seconds': time.perf_counter() - start,
                       'checkpoint': str(checkpoint),
                       'denoising_steps':policy.model.action_head.num_inference_timesteps, 'ensemble':ensemble})
    except EOFError:
        pass
    except Exception:
        traceback.print_exc()
        conn.send({'error': traceback.format_exc()})
    finally:
        conn.close()
listener.close()
