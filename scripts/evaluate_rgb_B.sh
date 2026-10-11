#!/usr/bin/env bash
# Real N1.7 inference; no expert trajectory is used by this evaluation.
set -euo pipefail
cd /workspace/GR00T_DualArm_Project
source scripts/project_env.sh
: "${B_CHECKPOINT:=checkpoints/lift_pot_rgb_B/checkpoint-2000}"
: "${B_START_SEED:=200000}"
: "${B_EPISODES:=20}"
: "${B_EXECUTE_HORIZON:=5}"
: "${B_DENOISING_STEPS:=4}"
: "${B_TAG:=recovery_B_h5_d4}"
: "${B_PORT:=6007}"
# Start serve_gr00t_local.py with the same checkpoint and port separately.
.venvs/robotwin/bin/python scripts/evaluate_lift_pot_gr00t.py \
  --checkpoint "$B_CHECKPOINT" --port "$B_PORT" \
  --start-seed "$B_START_SEED" --episodes "$B_EPISODES" \
  --max-actions 200 --execute-horizon "$B_EXECUTE_HORIZON" \
  --denoising-steps "$B_DENOISING_STEPS" --ensemble 1 \
  --control fixed --physics-steps-per-action 15 --tag "$B_TAG"
