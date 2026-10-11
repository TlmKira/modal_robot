#!/usr/bin/env bash
# Prepared proposal only. User requires confirmation for estimated >1h GPU work.
set -euo pipefail
: "${APPROVED_RGB_B_TRAINING:?Set only after explicit user approval of configs/lift_pot_rgb_B_proposal.json}"
cd /workspace/GR00T_DualArm_Project
source scripts/project_env.sh
if [[ -e /tmp/gr00t_rgb_B_training ]]; then
  echo 'Output exists; preserving prior run.' >&2
  exit 1
fi
.venvs/gr00t/bin/python simulation/Isaac-GR00T/gr00t/experiment/launch_finetune.py \
  --base-model-path "$PROJECT_ROOT/checkpoints/GR00T-N1.7-3B" \
  --dataset-path "$PROJECT_ROOT/datasets/lerobot/lift_pot_train_rgb_v23_terminal_pad" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "$PROJECT_ROOT/configs/aloha_gr00t_modality.py" \
  --output-dir /tmp/gr00t_rgb_B_training \
  --global-batch-size 4 --gradient-accumulation-steps 4 \
  --dataloader-num-workers 2 --shard-size 256 --num-shards-per-epoch 4 \
  --max-steps 2000 --save-steps 500 --save-total-limit 1 --save-only-model --no-use-wandb
