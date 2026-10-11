"""Load a real saved checkpoint and predict from a held-out real episode."""
import argparse
import json
from pathlib import Path
import runpy
import time
import numpy as np
import torch
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.policy.gr00t_policy import Gr00tPolicy

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output-prefix", default="short_checkpoint")
parser.add_argument('--dataset', default='datasets/lerobot/lift_pot_validation_v21')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
config = runpy.run_path(str(root / "configs/aloha_gr00t_modality.py"))["config"]
loader = LeRobotEpisodeLoader(root / args.dataset, config)
episode = loader[0]
observation = {
    "video": {key: np.asarray(episode[f"video.{key}"].iloc[0], dtype=np.uint8)[None, None]
              for key in config["video"].modality_keys},
    "state": {key: np.asarray(episode[f"state.{key}"].iloc[0], dtype=np.float32).reshape(1, 1, -1)
              for key in config["state"].modality_keys},
    "language": {"annotation.human.task_description": [[str(episode["language.annotation.human.task_description"].iloc[0])]]},
}
policy = Gr00tPolicy(embodiment_tag="NEW_EMBODIMENT", model_path=args.checkpoint, device="cuda:0")
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start = time.perf_counter()
actions, _ = policy.get_action(observation)
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
assert all(np.isfinite(value).all() for value in actions.values())
assert [actions[key].shape for key in config["action"].modality_keys] == [(1, 40, 6), (1, 40, 1), (1, 40, 6), (1, 40, 1)]
result = dict(checkpoint=str(Path(args.checkpoint).resolve()), validation_episode=0,
    finite_predictions=True, shapes={key: list(value.shape) for key, value in actions.items()},
    first_call_latency_seconds=elapsed, peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
    limitation="Single held-out forward pass; no closed-loop task success or validation-loss claim")
(root / f"evaluation/{args.output_prefix}_policy_check.json").write_text(json.dumps(result, indent=2))
np.savez(root / f"evaluation/{args.output_prefix}_validation_actions.npz", **actions)
print(result, flush=True)
