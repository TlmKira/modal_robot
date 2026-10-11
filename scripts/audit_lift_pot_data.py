"""Audit source trajectories before conversion; do not assume action semantics."""
import json
from pathlib import Path
import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "datasets/lift_pot/aloha-agilex_clean_50"
rows = []
for path in sorted(source.glob("data/*.hdf5")):
    with h5py.File(path) as f:
        q = f["joint_action/vector"][:]
        assert q.ndim == 2 and q.shape[1] == 14 and np.isfinite(q).all()
        for name, region in [("left_arm", slice(0, 6)), ("left_gripper", 6),
                             ("right_arm", slice(7, 13)), ("right_gripper", 13)]:
            assert np.allclose(q[:, region], f[f"joint_action/{name}"][:])
        cameras = {}
        for name, camera in f["observation"].items():
            assert len(camera["rgb"]) == len(q)
            for index in range(len(q)):
                image = cv2.imdecode(np.frombuffer(camera["rgb"][index], np.uint8), cv2.IMREAD_COLOR)
                assert image is not None, (path, name, index)
            cameras[name] = list(image.shape)
        instructions = json.loads((source / "instructions" / f"{path.stem}.json").read_text())
        assert instructions["seen"] and instructions["unseen"]
        rows.append(dict(episode=path.stem, frames=len(q), camera_shapes=cameras,
            gripper_min=q[:, [6, 13]].min(0).tolist(), gripper_max=q[:, [6, 13]].max(0).tolist()))
report = dict(episodes=len(rows), total_frames=sum(r["frames"] for r in rows),
    vector_order=["left_arm_6", "left_gripper", "right_arm_6", "right_gripper"],
    checks="finite vectors; exact left/right slices; all RGB decode; matching frame counts; language",
    limitations=["No explicit timestamp dataset found", "Command versus measured state alignment pending",
        "Success metadata and scene split audit pending", "Model loader and normalization checks pending"],
    trajectories=rows)
(ROOT / "evaluation/lift_pot_data_audit.json").write_text(json.dumps(report, indent=2))
print({k: v for k, v in report.items() if k != "trajectories"})
