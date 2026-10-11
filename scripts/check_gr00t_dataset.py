"""Load real converted episodes using the official GR00T episode loader."""
import json
import runpy
from pathlib import Path
import numpy as np
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

ROOT = Path(__file__).resolve().parents[1]
config = runpy.run_path(str(ROOT / "configs/aloha_gr00t_modality.py"))["config"]
results = []
for split in ["train", "validation", "test"]:
    loader = LeRobotEpisodeLoader(ROOT / f"datasets/lerobot/lift_pot_{split}_v21",
                                  config)
    frame = loader[0]
    for key in ["left_arm", "left_gripper", "right_arm", "right_gripper"]:
        for modality in ["state", "action"]:
            values = np.stack(frame[f"{modality}.{key}"].values)
            assert np.isfinite(values).all()
            assert values.shape[-1] == (1 if "gripper" in key else 6)
    for camera in ["head", "left_wrist", "right_wrist"]:
        values = np.stack(frame[f"video.{camera}"].values)
        assert values.shape[1:] == (240, 320, 3)
    result = dict(split=split, episodes=len(loader), sampled_episode_frames=len(frame),
                  columns=list(frame.columns))
    results.append(result)
    print(result, flush=True)
(ROOT / "evaluation/gr00t_dataset_loader_check.json").write_text(json.dumps(results, indent=2))
