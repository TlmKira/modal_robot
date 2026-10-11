# Framework version audit — 2026-10-10 UTC

Read-only official GitHub API and PyPI queries; raw responses: `evaluation/framework_upstream_audit.json`, `framework_related_changes.json`. No repository checkout or Python package upgrades performed.

| Framework | Current | Latest official release / commit | Decision |
|---|---|---|---|
| Isaac-GR00T | `d2b7e75b937e3ec9aa5dbc798f08b89692c49734` | n1.7-release (2026-04-18T14:45:14Z); tag commit 23ace64f17aa5015259b8609d371eb61a357c776 | Keep: local equals latest main HEAD; N1.7 checkpoint loads. Tagged release is Early Access, not a newer fix. |
| RoboTwin | `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f` | release (2026-02-26T20:18:14Z); tag commit bf44be51cf5717a5595ce59447f2cf5263d2aa95 | Keep: local equals latest main HEAD; official expert replay succeeds. |
| XPolicyLab | `fa431ecd893ee706883e64fe5fe1464ec8cd928d` | No published release or tags | Keep pinned RoboTwin submodule; latest changes are not a validated migration for this checkpoint. |
| LeRobot | `installed LeRobot 0.3.3; no independent LeRobot Git checkout` | v0.6.1 (2026-08-03T14:19:42Z); tag commit 7e241bd630a3719a56157a497ce5d08f244784f1 | Keep conversion baseline 0.3.3/v2.1. Trial 0.6.1 only in a separate environment; v3 data and new policy/processor APIs need migration tests. |

## Main branch inventory

- Isaac-GR00T: [d2b7e75b937e3ec9aa5dbc798f08b89692c49734](https://github.com/NVIDIA/Isaac-GR00T/commit/d2b7e75b937e3ec9aa5dbc798f08b89692c49734) — 2026-10-07T22:35:16Z; Sync GR00T N1.7 staging package to Isaac-GR00T.
- RoboTwin: [ea8b21121ebb3cd201ff5b3fe361944ac94eda3f](https://github.com/RoboTwin-Platform/RoboTwin/commit/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f) — 2026-09-24T03:44:58Z; Merge pull request #512 from FlowsMple/main.
- XPolicyLab: [896edbfc0530f192a2bfbd86d966d5f9bd0f5a43](https://github.com/XPolicyLab/XPolicyLab/commit/896edbfc0530f192a2bfbd86d966d5f9bd0f5a43) — 2026-10-10T04:17:26Z; [policy] Rename liber_0_lite to Liber_0_lite.
- LeRobot: [b9cb121cb7d4e3e26ec5c906d08dda68d151ba52](https://github.com/huggingface/lerobot/commit/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52) — 2026-10-08T21:24:13Z; perf(datasets): run image transforms on the GPU with image_transforms.backend=gpu (#4627).

## Verified changes and known issues

- Isaac-GR00T already includes the July 29 inference image-path change (`238ef45`), June shard-loading fix (`026af18`), and action-horizon validation fix (`376ba89`). Current main is N1.7; no model family switch is needed. The separately announced N1.7 General Release is commit `1a1837f205` (2026-07-07); the latest GitHub release tag remains the older Early Access tag. Main branch commits are not automatically stable releases. [Official history](https://github.com/NVIDIA/Isaac-GR00T/commits/main/)
- RoboTwin already includes reproducible episode/model enumeration fixes (#509/#490), joint-limit checking (`bd56368`) and codec synchronization. These improve reproducibility and image correctness; they do not certify policy success. [Official history](https://github.com/RoboTwin-Platform/RoboTwin/commits/main/)
- XPolicyLab RGB handling was unified by [`66ecde3`](https://github.com/XPolicyLab/XPolicyLab/commit/66ecde3ed34a83e4de56c7a36a3e81fe29f093ec), 2026-08-04; adapter/converter audit completed by [`14bb308`](https://github.com/XPolicyLab/XPolicyLab/commit/14bb3081611b093227b3cd3262bb72d39824161a), 2026-09-10. Both precede the pinned submodule. The project’s own historical conversion still added an extra swap, so an upstream update cannot repair existing videos/checkpoints.
- [N1.7 issue #742](https://github.com/NVIDIA/Isaac-GR00T/issues/742): padded action noise may affect valid predictions. [PR #741](https://github.com/NVIDIA/Isaac-GR00T/pull/741) is OPEN and unmerged at audit time. Applying a new mask only at inference changes the distribution for existing weights; no unverified patch applied. This is a potential mechanism, not a demonstrated cause here.
- [Issue #745](https://github.com/NVIDIA/Isaac-GR00T/issues/745) concerns the unused `use_mean_std` flag. This checkpoint sets it false; actions use recorded percentile/min-max statistics. No evidence that this issue caused the observed failures.
- [Issue #546](https://github.com/NVIDIA/Isaac-GR00T/issues/546) and [official data configuration](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/data_config.md): changing model action horizon requires matching relative statistics. We keep prediction horizon 40 and only change the executed prefix.
- [LeRobot v0.6.1](https://github.com/huggingface/lerobot/releases/tag/v0.6.1), 2026-08-03, supports updated N1.7 integration, but includes a module rename and follows the v3 dataset migration. It is not a drop-in replacement for NVIDIA Gr00tPolicy checkpoints and this v2.1 converter.

## SAPIEN and actual environment

- SAPIEN installed: 3.0.0b1; latest PyPI stable: 3.0.3, uploaded 2026-03-10. Official 3.0.3 tag maps to `731622eac5b140b320076c8a1b6eb4b553c3ccd4`; binary wheel provenance is not independently attested. Official tag mapping is recorded in `framework_related_changes.json`. [Official docs](https://sapien-sim.github.io/docs/), [PyPI metadata](https://pypi.org/pypi/sapien/json), [source tags](https://github.com/haosulab/SAPIEN/tags).
- Keep 3.0.0b1 for the recovery baseline because it is the RoboTwin pinned requirement and expert physics replay passes. 3.0.3 must first pass scene, camera, planner, drive and expert-replay comparisons in an isolated environment; physics changes can alter grasp contact outcomes.
- Existing GR00T: Python 3.12 / torch 2.9.0+cu128 / transformers 4.57.3 / numpy 1.26.4 / pyarrow 20.0.0 / flash-attn 2.8.3. Simulation: Python 3.10 / torch 2.4.1 / SAPIEN 3.0.0b1. Conversion: LeRobot 0.3.3 / pyarrow 25.0.1. Actual metadata captured in `configs/recovery_*_versions.json`.
- Container recreation removed EGL/GL user libraries although persistent virtualenvs survived. Restored only container user libraries `libegl1 libgl1 libglvnd0 vulkan-tools`; SAPIEN now creates a renderer and executes the official task. No Python environment overwritten.

## Storage and checkpoint preservation

- `/`: overlay Container Disk, 650GB, temporary container storage. `/workspace`: `mfs#eu-se-1.runpod.net:9421` FUSE network volume, separate mount. Project artifacts survived this recreation. This supports persistence across the observed recreation, not a guarantee of retention after volume deletion.
- `df` 756TB/235TB is shared backend capacity. No `quota` utility/account quota endpoint is available in this session; precise personal quota remains UNVERIFIED. Do not infer it from backend free space.
- Existing checkpoint/data/videos preserved. No external backup destination configured. Header validation and configuration hashes recorded in `evaluation/recovery_asset_audit.json`; checkpoint tensor offsets and shard index agree. Full model reload is tested by actual inference.

## Project-specific sampling finding

Latest official main retains `length - action_horizon + 1` observation sampling. On this short Lift Pot dataset with horizon 40, this excludes 1560 real training observations and all 990 closed-gripper states. Merely upgrading dependencies or toggling allow_padding does not restore that sampler range. Verified training-only terminal future-target padding in a new dataset restores all 4444 real observations without modifying baseline upstream code. See `evaluation/training_tail_coverage_audit.json` and `terminal_pad_sampler_check.json`. [Official sampler](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/data/dataset/sharded_single_step_dataset.py).

## Migration acceptance criteria

Use a separate environment/checkouts; preserve all baseline artifacts. Require unchanged checkpoint tensor loading, processor/statistics and 14-dimensional action mapping; identical golden-observation preprocessing and decoded actions; expert replay success; then fixed-seed real-model evaluation. Recompute statistics only for a new dataset/config version. Do not change N1.7 model family or reinterpret joint targets as EEF deltas.
