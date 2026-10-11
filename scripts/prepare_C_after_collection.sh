#!/usr/bin/env bash
set -euo pipefail
cd /workspace/GR00T_DualArm_Project
source scripts/project_env.sh
.venvs/data_conversion/bin/python scripts/build_C_dataset.py
.venvs/gr00t/bin/python -m gr00t.data.stats --dataset-path datasets/lerobot/lift_pot_C_train_v1 --embodiment-tag NEW_EMBODIMENT --modality-config-path configs/aloha_gr00t_modality.py
python - <<'PY'
from pathlib import Path
import shutil
base=Path('datasets/lerobot')
for split in ['validation','test']:
 for name in ['stats.json','relative_stats.json']:
  shutil.copy2(base/'lift_pot_C_train_v1/meta'/name,base/f'lift_pot_C_{split}_v1/meta'/name)
PY
.venvs/gr00t/bin/python scripts/check_C_dataset.py
