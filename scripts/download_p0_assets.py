"""Download pinned official assets and only the initial task dataset."""
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

root = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(root / "cache/huggingface")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import hf_hub_download

revision = "3dc3b798668feb99ac61cc9086d84cbcc3d79186"
files = [
    ("embodiments.zip", "simulation/RoboTwin/assets"),
    ("objects.zip", "simulation/RoboTwin/assets"),
    ("dataset/lift_pot/aloha-agilex_clean_50.zip", "datasets/downloads"),
]
def download(item):
    name, dest = item
    path = hf_hub_download("TianxingChen/RoboTwin2.0", name,
        repo_type="dataset", revision=revision, local_dir=root / dest)
    print(path, flush=True)
with ThreadPoolExecutor(3) as pool:
    list(pool.map(download, files))
