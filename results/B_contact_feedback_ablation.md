# B 接触反馈与执行块受控诊断

## 实测结论（15/15 完成）

**现有 B 已在未训练的 base0 场景通过官方 Lift Pot 判定：完整实测反馈在 seed 19 成功；drive-state 基线在 seed 107 成功。没有训练、升级或改变 checkpoint。完整反馈没有提高这 3 个配对场景的总成功数（两者均 1/3），而是交换了成功场景，不能认定为普适改进。**

| 方案 | 诊断成功 | 成功种子 |
|---|---:|---|
| drive_h5 | 1/3 | 107 |
| qpos_arm_h5 | 0/3 | 无 |
| qpos_full_h5 | 1/3 | 19 |
| drive_h1 | 0/3 | 无 |
| expert_replay | 3/3 | 107, 19, 48 |

1. **真实关节反馈改善跟踪，但不是充分条件。** 失败基线 seed 48 接触后左右 TCP 跟踪均差约 31/59 mm，seed 19 为 29/129 mm；仅实测臂关节使三场景接触后均差降至约 2.6–4.5 mm，但仍 0/3。完整反馈接触后约 2.7–5.0 mm，只有 seed 19 成功。这支持 command-state 造成接触后误差累积，也说明抓取定位与夹爪阶段仍是限制因素。

2. **缩短到 1 步未改善成功。** 同种子、同 drive 输入、同 60 ms 周期、同模型随机种子；各场景首块预测逐位相同（最大差 0），因此没有初始场景/模型加载混杂。1 步组全部失败，且接触后均差仍约 19–92 mm，不能假设更频繁重规划自然更好。该对比仅测试 drive 输入，不外推“完整实测反馈＋1 步”的效果；该组合未运行。推理中位数约 150 ms，1 步回合消耗更多推理墙钟时间。原动作的模拟时间未缩放，但当前推理速度不足以在真实硬件保证 60 ms 同步控制。

3. **首次接触时已经有定位偏差。** 专家三场景首次接触左右距离约 1.16–1.25 / 0.27–0.29 cm，接触后 TCP 跟踪约 2–3 mm，全部成功。失败 drive 基线 seed 48 首次接触为 4.32/6.48 cm，seed 19 为 2.74/6.04 cm，均先出现至少一侧超 3 cm，再发生跟踪发散。1 步组首次接触右臂均为 5.56–6.74 cm。仅实测反馈可消除大部分后期执行偏差，却常停留在错误抓取位置。首次碰到锅并非成功握住指定把手；距离条件也不是任何首次接触必需的阈值，基线 seed 107 初次接触较远，随后仍能靠近并成功。

4. **夹爪尺度与语义已核验。** 输入反馈为物理 base 指关节按控制 scale 反算的命令等价值，两个实际指关节米值另存。机械 q=0 对应等价值 0.181818；此时可已到闭合下限。专家 seed 48 成功时命令 0、实测等价值约 0.239，说明物体阻挡与负值驱动死区使物理状态无法与训练命令逐项等价。seed 19 实测方案的成功不是“归一化已消除语义差异”的证明；它只说明该明确记录的替换在这个场景有效。百分位归一化沿用原 processor，没有重算统计、改变 action 的相对/绝对定义或叠加状态。

5. **覆盖不足是可能的残余因素，尚未证实唯一根因。** 本组没有 base1 模型差异，三条匹配专家均成功，故物体模型未见/场景不可行不能解释本组失败。模型出现不同于示范的碰撞、半闭合和停滞状态，且多组有明显状态百分位范围超界；支持闭环状态覆盖不足的判断，但没有训练对照验证，不能直接宣布必须加数据或加训练步数。只读取测试划分用于本次诊断，样本少且选自专家可行场景，不能混入先前 0/20 或作为新的正式成功率。

## 两次真实模型成功的核验

| 场景 / 方案 | 动作数 | 锅高度 m | 左/右抓取距 cm | 锅朝上 dot | 双指接触累计 s |
|---|---:|---:|---|---:|---:|
| 107 / drive_h5 | 87 | 0.820156 | 2.735/2.832 | 0.9506 | 0.128 |
| 19 / qpos_full_h5 | 115 | 0.820040 | 2.844/2.547 | 0.9834 | 1.348 |

[优先查看 seed 19 完整实测反馈成功视频](../videos/B_contact_feedback_qpos_full_h5_seed19_raw.mp4)：结束时两侧指尖—锅接触均为真，累计双侧接触 1.348 s；模型真实输出并执行，未用专家替代。seed 107 基线也通过官方判定，累计双侧接触 0.128 s，但末帧右侧没有非零指尖接触冲量，不能据此宣称持续双臂稳固握持。两者均按首次官方成功立即结束，未另行验证成功后长时间保持。

15 个原始视频均已 ffprobe 和全片解码检查；成功来源、配置、物理状态、动作日志及视频 SHA256 保存在 `evaluation/B_contact_feedback/success_provenance.json`。B checkpoint 的 10 个 JSON 文件与训练完成时持久复制校验一致；未写入模型目录。原 20 次正式评测仍为 0/20，未重复启动。

初期在 seed 200000 尝试新专家规划失败，未启动该场景模型回合；预检日志/失败记录保留，排除于本表。随后按用户“已有数据对应”条件改用原始测试划分 seed 48/19/107，直接读取相同源专家轨迹。

**下一步判断：无需新训练即可获得真实未见 base0 成功，但尚无在这 3 场景稳定胜出的统一状态/执行方案。保留 drive＋5 步作为兼容基线及完整反馈＋5 步作为受控候选；不按种子切换方案冒充统一策略。当前不启动新训练、升级或完整 20 次重测。**

以下为方法、逐回合指标、曲线与视频。

已记录 15 个小规模诊断回合；不重复原 20 次正式评测。checkpoint 固定为 `checkpoints/lift_pot_rgb_B/checkpoint-2000`，GR00T N1.7，无训练/升级。

种子为 48、19、107：均为原始数据测试划分的未训练 base0 场景。每个场景读取对应原始官方专家轨迹，通过相同固定控制接口回放，专家结果不计模型成功。各组启动时物体 pose 逐项验证一致；模型随机种子同场景一致。

训练状态是 drive target，不是实际 qpos。实测臂 qpos 保持 rad 与相同关节顺序，仍使用 B 原有 q01/q99 归一化、相对动作解码；不改统计。`qpos_arm_h5` 只更换 12 个臂关节，夹爪输入保留命令；`qpos_full_h5` 再用物理夹爪关节反算实测开合量。夹爪 base/mimic 各关节按 (q-offset)/multiplier、URDF scale [-0.01,0.045] 反算，保留 base/mimic 两指原始读数，以 base joint 反算值作为单维模型反馈，输入裁剪到 [0,1]；URDF 指关节下限 0 m、上限 0.04765 m，而命令 scale 下限为 -0.01 m；因此物理闭合至 q=0 时，命令尺度等价值为 0.181818，绝不能解释成仍打开 18%。另记录 base/mimic 实测米值和下限状态。握住物体时不能仅凭未完全到机械下限判失败。物理 qpos 与训练 command 状态有分布差别，夹爪物理值不能表达示范中以命令为状态的同一控制阶段，本实验评估该替换效果，不宣称语义等价。超出训练 q01/q99 的比例以及超过范围宽度 1% 的显著超界比例写入 CSV，微小边界噪声不能单凭计数判作严重分布外。state 夹爪 q99=1.0，action 夹爪 q99=0.96231，两者不能混淆。

动作时间始终是 15×4 ms=60 ms。执行块 5/1 对比只使用 drive 输入，重规划的仿真间隔从 300 ms 改为 60 ms。推理期间不推进物理时钟；因此仿真时序保持一致，但更高推理负担不能声称适用于真实硬件实时 60 ms 控制。执行器插值起点始终为 drive target，未同时更改控制起点。

接触定义：锅 link 与左右臂/指尖 link 的接触 impulse 总量 >1e-8 Ns，排除锅与桌面。逐物理步检测首次接触及接触持续时间；每动作及首次事件保存遥测。FK(real qpos) 与原生 TCP 必须误差 <0.1 mm，否则停止记录为错误；不以接触点距离替代物理接触。

| 种子 | 方案 | 官方成功 | 最佳同时 max(L,R) cm | 终点左/右距离 cm | 首次机械臂接触 s | 推理均时 ms |
|---:|---|---|---:|---|---:|---:|
| 107 | drive_h1 | False | 5.23 | 36.88/25.29 | 6.740 | 133.893 |
| 19 | drive_h1 | False | 6.35 | 33.79/12.96 | 5.024 | 135.048 |
| 48 | drive_h1 | False | 6.46 | 15.16/29.81 | 5.072 | 152.580 |
| 107 | drive_h5 | True | 2.63 | 2.74/2.83 | 3.356 | 151.099 |
| 19 | drive_h5 | False | 6.04 | 4.38/16.91 | 3.892 | 152.894 |
| 48 | drive_h5 | False | 4.11 | 25.65/16.02 | 3.764 | 241.884 |
| 107 | expert_replay | True | 0.76 | 0.76/0.33 | 5.012 | — |
| 19 | expert_replay | True | 0.82 | 0.82/0.32 | 5.116 | — |
| 48 | expert_replay | True | 0.90 | 0.90/0.27 | 4.864 | — |
| 107 | qpos_arm_h5 | False | 3.15 | 36.49/18.87 | 5.516 | 151.993 |
| 19 | qpos_arm_h5 | False | 2.38 | 1.87/4.63 | 4.832 | 152.690 |
| 48 | qpos_arm_h5 | False | 5.09 | 14.07/6.63 | 4.552 | 154.807 |
| 107 | qpos_full_h5 | False | 4.61 | 7.26/6.34 | 5.612 | 142.560 |
| 19 | qpos_full_h5 | True | 2.84 | 2.84/2.55 | 5.180 | 152.056 |
| 48 | qpos_full_h5 | False | 5.09 | 6.00/5.08 | 4.552 | 156.008 |

分组计数：`{"drive_h1": {"episodes": 3, "successes": 0}, "drive_h5": {"episodes": 3, "successes": 1}, "expert_replay": {"episodes": 3, "successes": 3}, "qpos_arm_h5": {"episodes": 3, "successes": 0}, "qpos_full_h5": {"episodes": 3, "successes": 1}}`。3 个配对诊断种子不是新的 20 次正式成功率，用于配对诊断，不能视为新的完整 20 种子盲测。

逐回合指标见 [metrics CSV](B_contact_feedback_metrics.csv)，时间序列见 [timeseries CSV](B_contact_feedback_timeseries.csv)。详细物理夹爪、接触、预测/实际命令、checkpoint 与配置日志保存在 `evaluation/B_contact_feedback/`，完整运行日志 `logs/B-contact-feedback-data-seeds.log`。

![接触距离、跟踪偏差及锅高度](B_contact_feedback_curves.png)

![左右抓取距离和目标—实际 TCP 跟踪](B_contact_feedback_LR_curves.png)

## 原始真实诊断视频

- [drive_h1 / seed 107 / success=False](../videos/B_contact_feedback_drive_h1_seed107_raw.mp4)
- [drive_h1 / seed 19 / success=False](../videos/B_contact_feedback_drive_h1_seed19_raw.mp4)
- [drive_h1 / seed 48 / success=False](../videos/B_contact_feedback_drive_h1_seed48_raw.mp4)
- [drive_h5 / seed 107 / success=True](../videos/B_contact_feedback_drive_h5_seed107_raw.mp4)
- [drive_h5 / seed 19 / success=False](../videos/B_contact_feedback_drive_h5_seed19_raw.mp4)
- [drive_h5 / seed 48 / success=False](../videos/B_contact_feedback_drive_h5_seed48_raw.mp4)
- [expert_replay / seed 107 / success=True](../videos/B_contact_feedback_expert_replay_seed107_raw.mp4)
- [expert_replay / seed 19 / success=True](../videos/B_contact_feedback_expert_replay_seed19_raw.mp4)
- [expert_replay / seed 48 / success=True](../videos/B_contact_feedback_expert_replay_seed48_raw.mp4)
- [qpos_arm_h5 / seed 107 / success=False](../videos/B_contact_feedback_qpos_arm_h5_seed107_raw.mp4)
- [qpos_arm_h5 / seed 19 / success=False](../videos/B_contact_feedback_qpos_arm_h5_seed19_raw.mp4)
- [qpos_arm_h5 / seed 48 / success=False](../videos/B_contact_feedback_qpos_arm_h5_seed48_raw.mp4)
- [qpos_full_h5 / seed 107 / success=False](../videos/B_contact_feedback_qpos_full_h5_seed107_raw.mp4)
- [qpos_full_h5 / seed 19 / success=True](../videos/B_contact_feedback_qpos_full_h5_seed19_raw.mp4)
- [qpos_full_h5 / seed 48 / success=False](../videos/B_contact_feedback_qpos_full_h5_seed48_raw.mp4)

15/15 回合已完成；其中 12 次模型执行、3 次独立标记的专家回放。成功回合提前结束，失败为 200 动作上限，各组后接触时长不同，平均跟踪误差属于描述性统计。复现：先用现有 GR00T 环境启动 `serve_gr00t_local.py --checkpoint checkpoints/lift_pot_rgb_B/checkpoint-2000 --port 6009`，再用现有 RoboTwin 环境运行 `B_contact_feedback_runner.py --port 6009`。runner 拒绝覆盖已有结果/视频，复现时 runner 和汇总均传入相同的新 `--run-label replay01` 以保存独立版本；不可直接重跑覆盖。汇总：`.venvs/gr00t/bin/python scripts/summarize_B_contact_feedback.py`。
