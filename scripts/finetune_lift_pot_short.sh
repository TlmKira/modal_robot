#!/usr/bin/env bash
# Prepared command only; do not invoke until Cosmos access and forward pass are verified.
set -euo pipefail
cd /workspace/GR00T_DualArm_Project
source scripts/project_env.sh
source .venvs/gr00t/bin/activate
python simulation/Isaac-GR00T/gr00t/experiment/launch_finetune.py \
  --base-model-path "$PROJECT_ROOT/checkpoints/GR00T-N1.7-3B" \
  --dataset-path "$PROJECT_ROOT/datasets/lerobot/lift_pot_train_v21" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$PROJECT_ROOT/configs/aloha_gr00t_modality.py" \
  --output-dir "$PROJECT_ROOT/checkpoints/lift_pot_short" \
  --global-batch-size 4 --gradient-accumulation-steps 4 \
  --dataloader-num-workers 2 --shard-size 256 --num-shards-per-epoch 4 \
  --max-steps 100 --save-steps 50 --no-use-wandb
