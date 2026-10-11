# GR00T recovery closed-loop tests — 2026-10-10

All rows use live checkpoint-2000 predictions. Expert replay is diagnostic only. Baseline historical 5 independent seeds all failed; earlier runs mixed control settings and are not a fixed-policy success estimate.

| Tag | Seed | Train scene diagnostic | Prefix | Physics ticks | Measured state/control start | Velocity cap | Denoising/ensemble | Success | Actions | Error |
|---|---:|---|---:|---:|---|---|---|---|---:|---|
| recovery_h1_train | 1 | True | 1 | 15 | False/False | None | 4/1 | False | 200 |  |
| recovery_h40_train | 1 | True | 40 | 15 | False/False | None | 4/1 | False | 200 | RuntimeError('Pot fell below table; stopping failed rollout') |
| recovery_h40_unseen | 200000 | False | 40 | 15 | False/False | None | 4/1 | False | 200 |  |
| recovery_h40_unseen | 200001 | False | 40 | 15 | False/False | None | 4/1 | False | 160 | RuntimeError('Pot fell below table; stopping failed rollout') |
| recovery_h5_measured_train | 1 | True | 5 | 15 | True/False | None | 4/1 | False | 200 |  |
| recovery_h5_real_feedback_train | 1 | True | 5 | 15 | True/True | None | 4/1 | False | 80 | RuntimeError('Pot fell below table; stopping failed rollout') |
| recovery_h5_train | 1 | True | 5 | 15 | False/False | None | 4/1 | False | 200 |  |
| recovery_h5_true_d16_train | 1 | True | 5 | 15 | False/False | None | 16/1 | False | 200 |  |
| recovery_h5_v2_train | 1 | True | 5 | 15 | False/False | 2.0 | 4/1 | False | 200 |  |

Expert original episode 1 / seed 1 replay: official success; 105 targets; not counted above.

Predictions, executed targets and observed states are saved per rollout as NPZ; per-call tracking/grasp distances in JSON; raw simulation videos retained. Results should be grouped by one fixed tag/configuration. Training scenes are excluded from independent formal rates. Repeated seeds for factor comparisons are paired diagnostics, not new independent test seeds.

Recorded completed recovery rollouts: 9; successes 0. This total mixes diagnostics/configurations and is not a formal success rate.

No success delivery video is named `videos/01_lift_pot_gr00t.mp4` unless an actual recorded model rollout passes the official check.

## 尚未完成

本轮 7 个训练场景诊断与 2 个独立 seed 评测均失败；训练场景不计正式成功率。相同 h40 配置下独立评测为 0/2，样本量不足以宣称稳定成功率。尚未达到一次真实模型成功，因此未生成成功命名视频，20 种子正式验证和多任务扩展未完成。以上为 A 诊断结果。用户随后回复 `ok` 批准 B；B 已按独立配置启动，尚在训练，结果另行记录。

## Reproduction

```bash
source scripts/project_env.sh
.venvs/robotwin/bin/python scripts/replay_lift_pot_interface.py --seed 1 --source-episode 1
.venvs/gr00t/bin/python scripts/offline_recovery_metrics.py
.venvs/robotwin/bin/python scripts/audit_live_source_scene.py
.venvs/gr00t/bin/python scripts/audit_processor_roundtrip.py
# Server in separate terminal:
.venvs/gr00t/bin/python -u scripts/serve_gr00t_local.py --denoising-steps 4
# Use a NEW tag to preserve videos:
.venvs/robotwin/bin/python -u scripts/evaluate_lift_pot_gr00t.py --start-seed 1 --episodes 1 --max-actions 200 --execute-horizon 5 --control fixed --match-legacy-training-channels --tag recovery_new
```

Graphics recovery (container-only): `apt-get install -y libegl1 libgl1 libglvnd0 vulkan-tools`. Existing Python virtualenvs are reused. Exact current versions in `configs/recovery_*_versions.json`.

## New data reproduction (creation refuses to overwrite existing versions)

```bash
source scripts/project_env.sh
# Builders are for a fresh destination; preserve current completed datasets.
.venvs/robotwin/bin/python scripts/build_correct_rgb_dataset.py
.venvs/data_conversion/bin/python scripts/build_rgb_terminal_pad_dataset.py
bash scripts/refresh_terminal_pad_stats.sh
.venvs/gr00t/bin/python scripts/check_terminal_pad_sampler.py
```

Only NEW v23 training statistics are recomputed; validation/test copy those training statistics. Pre-refresh caches remain preserved. Existing A metadata/weights/data are unchanged. The B proposal is prepared in `configs/lift_pot_rgb_B_proposal.json`; the user subsequently approved its estimated >1h GPU run, which is now running.


## B 后台实测结果（自动记录，待分析）

```json
{
  "checkpoint": "/workspace/GR00T_DualArm_Project/checkpoints/lift_pot_rgb_B/checkpoint-2000",
  "configuration": {
    "B_CHECKPOINT": "/workspace/GR00T_DualArm_Project/checkpoints/lift_pot_rgb_B/checkpoint-2000",
    "B_EXECUTE_HORIZON": "5",
    "B_DENOISING_STEPS": "4"
  },
  "requested_seeds": 20,
  "executed_seeds": 20,
  "successes": 0,
  "failures": 20,
  "initialization_or_inference_aborts": 0,
  "success_rate_executed": 0.0,
  "training_scene_excluded": true,
  "success_criterion": "RoboTwin lift_pot.check_success",
  "records": "evaluation/gr00t_lift_pot_recovery_B_h5_d4_test20.json"
}
```


成功命名视频现已保存，但来源为 seed 1 训练场景诊断，**不计入正式成功率**。详见 `evaluation/01_lift_pot_gr00t_provenance.json`。正式独立结果仍为 0/20。


### B 20 次结果进一步诊断（只读分析，未重复测试或训练）

详见 `results/lift_pot_B_test20_diagnosis.md`。20 次最后观测左右抓取距离均值 24.31/22.94 cm；其中两个掉落中止缺完整终点，18 次可用终点均值 19.02/16.99 cm。输出动作与下发 drive target 完全相同，B 实际 v23 的 150 帧三相机 RGB 抽检全部通过。真实关节跟踪误差早期约 1.5 mm，后期均值 125/172 mm；但 seed 200004 右 TCP 跟踪差仅 2.48 mm，命令 TCP 到抓取点仍至少差 60.78 mm，表明还存在模型抓取目标偏差。50 条原始示范均为 base0，独立测试 base0 12 次和 base1 8 次均失败。只测到一个训练场景成功，不能直接判定参数过拟合或把未见模型解释当成唯一根因。优先逐因素验证真实反馈/接触执行与短动作块，当前没有新增测试或训练；不建议重复已经完成的正确 RGB B 微调。


## B 接触反馈受控实验完成（2026-10-10）

在原始测试划分未训练 base0 种子 48、19、107 上完成 12 次真实模型执行和 3 次同场景源专家回放。固定 B / 60 ms 动作周期 / 去噪 4；drive+h5 1/3（107），实测臂关节+h5 0/3，实测臂关节和夹爪+h5 1/3（19），drive+h1 0/3，专家3/3。反馈显著减少接触后跟踪偏差，但未统一提高成功率，且定位偏差在首次接触已出现。完整反馈的 seed19 结束锅高度 0.820040 m、左右抓取距离 2.844/2.547 cm、朝上 dot 0.9834，两侧末帧指尖接触均存在。基线107也官方成功，但末帧右侧无非零指尖冲量，不声称持续稳定双臂握持。两次成功均为真实模型输出，未调用专家替代；本次未训练或升级，checkpoint保持。原20次正式结果仍0/20，不与本次小规模条件诊断混计。所有15视频全片解码验证通过。详见 results/B_contact_feedback_ablation.md、results/B_contact_feedback_metrics.csv、evaluation/B_contact_feedback/success_provenance.json。
