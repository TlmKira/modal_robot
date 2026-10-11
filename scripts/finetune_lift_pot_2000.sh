#!/usr/bin/env bash
# Prepared for review. Run only after explicit approval of the formal training.
set -euo pipefail
cd /workspace/GR00T_DualArm_Project
source scripts/project_env.sh
source .venvs/gr00t/bin/activate
# Warm-start weights from the validated 100-step checkpoint; start a fresh
# optimizer and 2000-step LR schedule. The short-run artifacts remain intact.
python simulation/Isaac-GR00T/gr00t/experiment/launch_finetune.py \
  --base-model-path "$PROJECT_ROOT/checkpoints/lift_pot_short/checkpoint-100" \
  --dataset-path "$PROJECT_ROOT/datasets/lerobot/lift_pot_train_v21" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$PROJECT_ROOT/configs/aloha_gr00t_modality.py" \
  --output-dir "$PROJECT_ROOT/checkpoints/lift_pot_2000" \
  --global-batch-size 4 --gradient-accumulation-steps 4 \
  --dataloader-num-workers 2 --shard-size 256 --num-shards-per-epoch 4 \
  --max-steps 2000 --save-steps 500 --save-only-model --no-use-wandb
