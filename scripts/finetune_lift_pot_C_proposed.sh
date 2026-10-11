#!/usr/bin/env bash
# Prepared only; run after the user explicitly confirms C training and its budget.
set -euo pipefail
[[ "${APPROVED_C_TRAINING:-}" == yes ]] || { echo 'C training requires explicit user confirmation.' >&2; exit 1; }
cd /workspace/GR00T_DualArm_Project
source scripts/project_env.sh
python - <<'CHECK_STORAGE'
import json
from pathlib import Path
s=json.loads(Path('evaluation/C_storage_preflight.json').read_text())
assert s.get('persistent_capacity_confirmed') is True, 'Personal persistent capacity must be confirmed before training'
CHECK_STORAGE
[[ -f evaluation/C_dataset_loader_check.json && -f results/C_data_quality/summary.json ]]
[[ ! -e /tmp/gr00t_C_v1_training && ! -e checkpoints/lift_pot_C_v1 ]]
.venvs/gr00t/bin/python scripts/launch_C_single_export.py \
 --base-model-path "$PROJECT_ROOT/checkpoints/GR00T-N1.7-3B" \
 --dataset-path "$PROJECT_ROOT/datasets/lerobot/lift_pot_C_train_v1" \
 --embodiment-tag NEW_EMBODIMENT \
 --modality-config-path "$PROJECT_ROOT/configs/aloha_gr00t_modality.py" \
 --output-dir /tmp/gr00t_C_v1_training \
 --global-batch-size 4 --gradient-accumulation-steps 4 \
 --learning-rate 0.0001 --dataloader-num-workers 2 --shard-size 256 \
 --num-shards-per-epoch 4 --max-steps 2000 --save-steps 500 \
 --save-total-limit 1 --save-only-model --no-use-wandb
