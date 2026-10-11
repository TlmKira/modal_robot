# CPU-only fixes, 2026-10-11

The archived baseline controller and interface are under `docs/baseline_2026-10-10/`. Original weights, datasets, action NPZ, raw JSON and MP4 are untouched.

Implemented:

- `control_audit.py`: reusable physical-time assertion, exact half-open holding interval, boolean success transitions. No model or simulator imports.
- `BC_blind_runner.py`: keep drive-state input, decoded absolute targets, 15 ticks/action and h5. Reject an incompatible timestep. Capture complete states at first success/loss/regain. Protect environment cleanup even when setup/recorder creation fails. Require a unique run label, so revised code cannot silently overwrite the historical evaluation.
- `BC_blind_telemetry.py`: distinguish drive commands, measured joint positions/velocity and measured grippers; save object/actual TCP poses and all four Lift Pot success-condition components at event ticks. Restore the scene step hook when closing.
- `gr00t_robotwin_interface.py`: reject non-uint8/non-HWC RGB, inconsistent camera shapes, bad state sizes and nonfinite values; camera order and action concatenation unchanged.
- `project_env.sh`: resolve root relative to the script (or use explicit PROJECT_ROOT) for migration.
- `train_external.py`: dry-run-first, independent output directory, explicit training invocation and storage confirmation; defaults match B/C, does not launch any training automatically.

CPU tests: six unittest cases pass. They cover tick windows, instantaneous success loss/recovery, timing mismatch, image/action boundaries, numerical equivalence of the old/new target executor and unchanged historical success counts (B 0/20; C 9/20; continuous 2s hold 6/20).

Not implemented: qpos replacement, target projection, additional physical forces, expert substitution, per-seed policy selection, or training hyperparameter changes. Graphics/event recording and cleanup with real SAPIEN still need GPU validation. These fixes do not establish improved task success rates.

The old background evaluator is a historical one-shot orchestrator; do not rerun it. Revised runner requires `--run-label`. Updated controller experiments must use new config/output paths and be labelled development diagnostics unless seeds were frozen before tuning.
