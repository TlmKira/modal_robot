# Restore on another GPU host

This guide prepares a separate host; no training or resource purchase was started on the CPU-only host. Keep B and C immutable.

## Code and dependencies

Clone this repository into any chosen root. Clone the frozen official repositories under `simulation/` (the project Git does not contain their large assets):

```bash
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T.git simulation/Isaac-GR00T
git -C simulation/Isaac-GR00T checkout d2b7e75b937e3ec9aa5dbc798f08b89692c49734
git -C simulation/Isaac-GR00T submodule update --init --recursive
git clone https://github.com/RoboTwin-Platform/RoboTwin.git simulation/RoboTwin
git -C simulation/RoboTwin checkout ea8b21121ebb3cd201ff5b3fe361944ac94eda3f
```

Restore the Python3.12 GR00T environment using pinned official installation instructions and compare to `configs/gr00t_packages.txt`. Simulation uses a separate Python3.10/SAPIEN3.0.0b1 environment; data conversion uses the third environment. Do not install all three dependency lists into one environment. Freeze files include editable/local dependencies; adapt local paths, not library versions blindly. GPU driver compatibility and FlashAttention build must be checked on the new host. Training alone does not require starting SAPIEN.

```bash
source scripts/project_env.sh
# PROJECT_ROOT is now derived from the script location.
python -m unittest discover -s tests -v
```

CPU tests need only NumPy. The optional historical result test is skipped if raw local evaluation JSON is absent.

## Transfer existing artifacts separately

Use a trusted authenticated file transfer from the persistent volume. Do not send credentials through README or Git commits. Transfer the following directory trees with their relative filenames intact:

1. `checkpoints/GR00T-N1.7-3B` official base, including processor/config/statistics files.
2. `datasets/lerobot/lift_pot_C_train_v1`, `lift_pot_C_validation_v1`, `lift_pot_C_test_v1`, including all parquet/video/meta files.
3. B/C checkpoints only if inference/comparison is required; they are not needed to train a new model from the base.
4. Robot embodiment/object assets required by RoboTwin only for future simulation.

Verify shard indices, file checksums and dataset manifest seeds after transfer. The historical test split is not a fresh blind test. Keep validation/test out of training statistics. C already has trained drive semantics and correct RGB; do not mix qpos fields or redo conversion in place.

## New isolated training run

The portable launcher is dry-run by default and imports no model. Substitute paths that exist on the new host:

```bash
source scripts/project_env.sh
python scripts/train_external.py \
  --base-model "$PROJECT_ROOT/checkpoints/GR00T-N1.7-3B" \
  --dataset "$PROJECT_ROOT/datasets/lerobot/lift_pot_C_train_v1" \
  --output "$PROJECT_ROOT/checkpoints/C_external_run01" \
  --python "$PROJECT_ROOT/.venvs/gr00t/bin/python"
```

Default parameters reproduce C: 2000 updates, microbatch4, accumulation4, LR1e-4, official tuning defaults. To actually train on your separately authorized GPU, append `--execute --persistent-quota-confirmed` after verifying **personal** output quota. Shared filesystem free space does not prove personal quota. Output must not exist. Model checkpoint rotation can temporarily hold two full checkpoints (approximately25GB on this model); allow an additional5GB. A final duplicate root export is suppressed. No B/C weights are overwritten. `save_only_model` cannot resume optimizer/scheduler state.

Effective batch32 can be tested via `--accumulation 8`, but label this a new ablation: compute/sample exposure per2000 updates doubles. Do not simultaneously change learning rate/state semantics. Any long training budget should be approved on the new host before execution.

Capture stdout/stderr, GPU telemetry and actual runtime/cost. Check processor/statistics and held-out action error after training; loss alone is insufficient. Periodic independent validation must be added deliberately, because the historical configuration disables it. No success-rate improvement is asserted by the CPU fixes.

## Future evaluation

The revised runner requires a unique `--run-label` and explicit config path. Historical base0 seeds are development diagnostics now, not new blind tests. Keep old results unchanged. This runner currently supports model selectors B/C; to evaluate a new checkpoint, prepare a distinct configuration with its C path and new seeds. Model server and runner must use the same checkpoint and port.

```bash
# On a GPU host only, after permission to run simulation:
.venvs/gr00t/bin/python -u scripts/serve_gr00t_local.py \
  --checkpoint checkpoints/C_external_run01/checkpoint-2000 --port 6011
# Separate terminal, with the corresponding new configuration:
.venvs/robotwin/bin/python scripts/BC_blind_runner.py \
  --model C --port 6011 --run-label external_dev01 \
  --config-path configs/EXPLICIT_NEW_EVALUATION.json
```

`EXPLICIT_NEW_EVALUATION.json` is a placeholder to create, not an existing test configuration. The pipeline remains model-driven drive_h5/60ms. Revised events record exact success-loss states. Validate this telemetry on a small GPU diagnostic before a new formal20-seed run. Never rerun the historical background orchestration to overwrite results.
