"""Run official Lift Pot expert and record actual simulator RGB frames."""
import json
import faulthandler
import os
import sys
import traceback
from pathlib import Path
import yaml
import imageio.v2 as imageio
import argparse
import numpy as np
import sapien

parser = argparse.ArgumentParser()
parser.add_argument("--start-seed", type=int, default=0)
parser.add_argument("--attempts", type=int, default=30)
parser.add_argument("--wide", action="store_true")
options = parser.parse_args()

ROOT = Path(__file__).resolve().parents[1]
faulthandler.dump_traceback_later(90, repeat=True)
REPO = ROOT / "simulation/RoboTwin"
os.chdir(REPO)
sys.path.insert(0, str(REPO))
from envs.lift_pot import lift_pot
print("Official Lift Pot task imported", flush=True)

args = yaml.safe_load((REPO / "env_cfg/task_config/demo_clean.yml").read_text())
robot_file = REPO / "assets/embodiments/aloha-agilex"
robot_config = yaml.safe_load((robot_file / "config.yml").read_text())
args.update(task_name="lift_pot", task_config="p0_expert", embodiment_name="aloha-agilex",
    left_robot_file=str(robot_file), right_robot_file=str(robot_file),
    left_embodiment_config=robot_config, right_embodiment_config=robot_config,
    dual_arm_embodied=True, need_plan=True, save_data=False,
    save_path=str(ROOT / "datasets/local_expert"), render_freq=0, save_freq=8)
attempts = []
prefix = "lift_pot_expert_wide" if options.wide else "lift_pot_expert"
for seed in range(options.start_seed, options.start_seed + options.attempts):
    task = lift_pot()
    raw = ROOT / f"videos/{prefix}_seed{seed:03d}_raw.mp4"
    if raw.exists():
        raise FileExistsError(f"Preserving existing recording: {raw}")
    writer = imageio.get_writer(raw, fps=250/8, codec="libx264", quality=8,
                               macro_block_size=1)
    frames = [0]
    presentation_camera = [None]
    def capture():
        task._update_render()
        if presentation_camera[0] is None:
            task.cameras.update_picture()
            rgb = task.cameras.get_rgb()["head_camera"]["rgb"]
        else:
            presentation_camera[0].take_picture()
            rgb = np.clip(presentation_camera[0].get_picture("Color")[..., :3] * 255, 0, 255).astype(np.uint8)
        writer.append_data(rgb)
        frames[0] += 1
    task._take_picture = capture
    try:
        task.setup_demo(seed=seed, now_ep_num=seed, **args)
        if options.wide:
            position = np.array([1.2, -1.6, 1.55])
            forward = np.array([0, -0.25, 0.78]) - position
            forward /= np.linalg.norm(forward)
            left = np.cross([0, 0, 1], forward)
            left /= np.linalg.norm(left)
            matrix = np.eye(4)
            matrix[:3, :3] = np.stack([forward, left, np.cross(forward, left)], axis=1)
            matrix[:3, 3] = position
            presentation_camera[0] = task.scene.add_camera("presentation", 960, 720, np.deg2rad(52), 0.1, 10)
            presentation_camera[0].entity.set_pose(sapien.Pose(matrix))
        capture()
        task.play_once()
        success = bool(task.plan_success and task.check_success())
        attempts.append(dict(seed=seed, success=success, frames=frames[0],
                             pot_position=task.pot.get_pose().p.tolist(), video=str(raw)))
        print(attempts[-1], flush=True)
    except Exception as exc:
        traceback.print_exc()
        attempts.append(dict(seed=seed, success=False, error=repr(exc), frames=frames[0]))
        # Installation failures should not consume 30 retries.
        if isinstance(exc, (ImportError, FileNotFoundError, NameError)):
            raise
    finally:
        writer.close()
        if hasattr(task, "scene"):
            task.close_env()
        audit_name = "expert_wide_attempts.json" if options.wide else "expert_attempts.json"
        (ROOT / "evaluation" / audit_name).write_text(json.dumps(attempts, indent=2))
    if attempts[-1]["success"]:
        source_name = "expert_wide_success_source.txt" if options.wide else "expert_success_source.txt"
        (ROOT / "videos" / source_name).write_text(str(raw))
        break
else:
    raise RuntimeError("No successful official expert episode in 30 seeds")
