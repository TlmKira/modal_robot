"""Bridge official legacy HDF5 to the current official XPolicyLab writer."""
import json
import os
import sys
from pathlib import Path
import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "simulation/RoboTwin"
sys.path.insert(0, str(REPO))
os.chdir(REPO)
from envs.utils.pkl2hdf5 import create_xpolicylab_hdf5
from data.decode_image_bit import decode_image_bit

source = ROOT / "datasets/lift_pot/aloha-agilex_clean_50"
dest = ROOT / "datasets/xpolicylab/demo_clean/lift_pot/aloha_agilex/data"
dest.mkdir(parents=True, exist_ok=True)
for path in sorted(source.glob("data/*.hdf5")):
    output = dest / path.name
    if output.exists():
        print("Already exists", output, flush=True)
        continue
    with h5py.File(path) as f:
        data = {group: {key: value[:] for key, value in f[group].items()}
                for group in ["joint_action", "endpose"]}
        data["observation"] = {}
        for name, camera in f["observation"].items():
            # Official codec distinguishes legacy channel-reversed JPEGs from
            # marked standard RGB JPEGs. Existing trained datasets are retained;
            # their historical swapped channels are matched at live inference.
            images = [decode_image_bit(bits) for bits in camera["rgb"]]
            data["observation"][name] = {"rgb": np.asarray(images),
                "intrinsic_cv": camera["intrinsic_cv"][:],
                "extrinsic_cv": camera["extrinsic_cv"][:],
                "cam2world_gl": camera["cam2world_gl"][:]}
    instruction = json.loads((source / "instructions" / f"{path.stem}.json").read_text())["seen"]
    # Follow official dataset convention (15 Hz); precise source sampling remains documented.
    create_xpolicylab_hdf5(data, str(output), instruction, frequency=15)
    print(output, flush=True)
link = REPO / "data/demo_clean"
if not link.exists():
    link.symlink_to(ROOT / "datasets/xpolicylab/demo_clean", target_is_directory=True)
