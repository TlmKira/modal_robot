#!/usr/bin/env bash
# Refresh only the NEW dataset's caches; never modify baseline stats or weights.
set -euo pipefail
cd /workspace/GR00T_DualArm_Project
source scripts/project_env.sh
python - <<'PY'
from pathlib import Path
import shutil
meta=Path('datasets/lerobot/lift_pot_train_rgb_v23_terminal_pad/meta')
for name in ['stats.json','relative_stats.json']:
    backup=meta/name.replace('.json','_pre_refresh.json')
    if not backup.exists(): shutil.copy2(meta/name,backup)
    (meta/name).write_text('{}\n')
PY
.venvs/gr00t/bin/python -m gr00t.data.stats \
  --dataset-path datasets/lerobot/lift_pot_train_rgb_v23_terminal_pad \
  --embodiment-tag NEW_EMBODIMENT --modality-config-path configs/aloha_gr00t_modality.py
python - <<'PY'
from pathlib import Path
import shutil
base=Path('datasets/lerobot')
for split in ['validation','test']:
    for name in ['stats.json','relative_stats.json']:
        shutil.copy2(base/'lift_pot_train_rgb_v23_terminal_pad/meta'/name,
                     base/f'lift_pot_{split}_rgb_v23_terminal_pad/meta'/name)
PY
