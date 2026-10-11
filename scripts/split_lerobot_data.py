"""Create disjoint trajectory splits, retaining source provenance and video links."""
import copy
import json
import random
import shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "datasets/lerobot/lift_pot_all50_v21"
info = json.loads((source / "meta/info.json").read_text())
assert info["total_episodes"] == 50, "Wait for existing conversion to finish"
episodes = [json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines()]
source_names = sorted((ROOT / "datasets/xpolicylab/demo_clean/lift_pot/aloha_agilex/data").glob("*.hdf5"))
seeds = [int(n) for n in (ROOT / "datasets/lift_pot/aloha-agilex_clean_50/seed.txt").read_text().split()]
order = list(range(50))
random.Random(20261010).shuffle(order)
manifest = {"split_seed": 20261010, "evaluation_scene_seeds": list(range(200000, 200020)),
            "source_sampling_note": "Official conversion uses 15Hz; legacy save_freq=15 at 250Hz implies nominal 16.67Hz plus boundary samples. Exact per-frame timestamps absent.", "splits": {}}
for name, selected in [("train", order[:40]), ("validation", order[40:45]), ("test", order[45:])]:
    target = ROOT / "datasets/lerobot" / f"lift_pot_{name}_v21"
    if (target / "meta/info.json").exists():
        raise FileExistsError(f"Preserving existing split: {target}")
    (target / "meta").mkdir(parents=True, exist_ok=True)
    (target / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    out_episodes = []
    provenance = []
    global_index = 0
    for new_index, old_index in enumerate(selected):
        episode = copy.deepcopy(episodes[old_index])
        frame = pd.read_parquet(source / f"data/chunk-000/episode_{old_index:06d}.parquet")
        frame["episode_index"] = new_index
        frame["index"] = range(global_index, global_index + len(frame))
        frame.to_parquet(target / f"data/chunk-000/episode_{new_index:06d}.parquet", index=False)
        for camera in ["cam_high", "cam_left_wrist", "cam_right_wrist"]:
            dest_dir = target / f"videos/chunk-000/observation.images.{camera}"
            dest_dir.mkdir(parents=True, exist_ok=True)
            video = source / f"videos/chunk-000/observation.images.{camera}/episode_{old_index:06d}.mp4"
            assert video.exists()
            (dest_dir / f"episode_{new_index:06d}.mp4").symlink_to(video)
        episode["episode_index"] = new_index
        out_episodes.append(episode)
        source_episode = int(source_names[old_index].stem.replace("episode", ""))
        provenance.append(dict(episode_index=new_index, source_episode=source_episode,
            source_seed=seeds[source_episode], converted_episode=old_index, frames=len(frame)))
        global_index += len(frame)
    split_info = copy.deepcopy(info)
    split_info.update(total_episodes=len(selected), total_frames=global_index,
                      total_videos=len(selected)*3, splits={"train":f"0:{len(selected)}"})
    (target / "meta/info.json").write_text(json.dumps(split_info, indent=2))
    (target / "meta/episodes.jsonl").write_text("".join(json.dumps(e)+"\n" for e in out_episodes))
    shutil.copy2(source / "meta/tasks.jsonl", target / "meta/tasks.jsonl")
    shutil.copy2(ROOT / "configs/aloha_modality.json", target / "meta/modality.json")
    manifest["splits"][name] = provenance
    print(name, len(selected), global_index, flush=True)
sets = [{e["source_seed"] for e in entries} for entries in manifest["splits"].values()]
assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
(ROOT / "configs/dataset_split_manifest.json").write_text(json.dumps(manifest, indent=2))
