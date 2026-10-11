# GR00T N1.7 + RoboTwin 双臂实验

工作目录：`/workspace/GR00T_DualArm_Project`。P0 已完成；P1 已完成 2000 步微调和真实 GR00T 闭环测试；尚无通过官方判定的模型成功视频。

## 已验证

- RTX A6000，49140 MiB；系统 Python 3.12.3、PyTorch 2.8.0+cu128，CUDA 可用。
- 容器 CPU 配额 7.65 核，内存上限 49,999,998,976 字节。
- NVIDIA Vulkan 可用：初次失败由缺少 `libEGL.so.1` 引起；安装容器用户态 `libegl1 libgl1 libglvnd0` 后修复。未修改宿主机驱动或内核。
- 独立仿真环境 `.venvs/robotwin`：Python 3.10，SAPIEN 3.0.0b1。
- SAPIEN 真实相机渲染图像：`results/screenshots/sapien_smoke.png`。
- 官方 Lift Pot ALOHA clean 数据：50 条轨迹，5604 帧。初步检查见 `evaluation/lift_pot_data_initial_audit.json`。
- 专家交付视频：`videos/01_lift_pot_expert.mp4`，960×720、30fps、6.566667 秒、197 帧，H.264。全片解码和开始/中段/结束画面检查通过，官方成功判定通过。
- seed 0 专家失败；seed 1 成功。宽视角 seed 1 再次成功，锅最终高度 0.88124 米。此项是环境验证，不能作为模型成功率或独立 20 次评测。
- LeRobot v2.1 转换 50/50 成功，shift 后 5554 帧。训练/验证/测试为 40/5/5 条、4444/556/554 帧，按源轨迹与 seed 隔离；归一化只用训练集计算。
- GR00T 官方加载器已读取三个划分的真实视频、状态、动作、语言；见 `evaluation/gr00t_dataset_loader_check.json`。
- GR00T 权重已下载，1031 个 tensor 的分片 header 与 index 一致。Python 3.12 独立环境中的 PyTorch 2.9.0+cu128、FlashAttention 2.8.3 和 GR00T 导入通过。

## 环境与执行

```bash
cd /workspace/GR00T_DualArm_Project
source scripts/project_env.sh
source .venvs/robotwin/bin/activate
python scripts/render_smoke.py
python scripts/download_p0_assets.py
# P0 可以只下载锅资产，免下载完整 3.7GB 物体库：
python scripts/download_lift_pot_objects.py
python scripts/record_lift_pot_expert.py
# 仅新录制使用，脚本会拒绝覆盖同名原始视频：
python scripts/record_lift_pot_expert.py --start-seed 1 --attempts 1 --wide
```

录制调用官方专家 `lift_pot.play_once()`，原始视频按 seed 保存；成功判定来自官方 `check_success()`。这些结果必须标记 **Expert Demonstration**。

仓库固定 commit 和环境检查保存在 `configs/environment_initial.json`；安装、Vulkan 排障和渲染日志在 `logs/`。官方数据仓库为 `TianxingChen/RoboTwin2.0`，固定 revision `3dc3b798668feb99ac61cc9086d84cbcc3d79186`。

## 数据处理与接口

```bash
source scripts/project_env.sh
.venvs/robotwin/bin/python scripts/audit_lift_pot_data.py
.venvs/robotwin/bin/python scripts/prepare_legacy_lift_pot.py
HF_LEROBOT_HOME="$PROJECT_ROOT/datasets/lerobot" .venvs/data_conversion/bin/python \
  simulation/RoboTwin/XPolicyLab/scripts/transform_lerobot_v21_format.py \
  demo_clean.lift_pot.aloha_agilex --repo_id NEW_DATASET_NAME --max_episode 50
# 已完成的划分请保留，不要重新运行覆盖。
.venvs/gr00t/bin/python scripts/check_gr00t_dataset.py
```

状态和动作顺序：左 6 关节、左夹爪、右 6 关节、右夹爪；关节为弧度，夹爪为官方归一化值。官方转换使用 `state=q[t]`、`action=q[t+1]`。关节在模型内部采用相对表示，夹爪采用绝对表示；模型解码后关节目标已转回绝对坐标。

`scripts/gr00t_robotwin_interface.py` 准备了观测映射、动作拼接、有限值与 URDF 限位验证，真实数据格式检查通过，但尚未运行模型闭环。官方执行接口是 `task.take_action(action, action_type="qpos")`；该接口含 TOPP，实际控制频率及碰撞检查必须另行验证。

## 安装记录与训练前提

原环境保持不变。仿真和训练依赖清单：`configs/robotwin_packages.txt`、`configs/gr00t_packages.txt`、`configs/conversion_packages.txt`。初始安装日志、cuRobo 0.7.8 构建日志均在 `logs/`。图形运行库修复命令为 `apt-get install -y vulkan-tools ffmpeg libegl1 libgl1 libglvnd0`，仅作用于容器。

训练依赖采用官方版本；暂未安装仅导出使用的 ONNX/TensorRT 组件。为兼容官方 NumPy 1.26.4，PyArrow 固定为 20.0.0（新版本 26.0.0 导入要求 NumPy 2）。

短训练命令 `bash scripts/finetune_lift_pot_short.sh` 已执行成功。Cosmos 和 GR00T 两模型访问授权均已验证，并已下载加载。不要重复执行短训练覆盖成果，也不要把访问 token 写入报告或对话。

训练后加载使用官方 `Gr00tPolicy(embodiment_tag="NEW_EMBODIMENT", model_path="实际 checkpoint 目录", device="cuda:0")`。当前基础模型不包含此新本体的已微调策略，不能宣称现有视频由 GR00T 控制。

## 尚未完成

当前已完成 2000 步微调、checkpoint 重载及真实闭环诊断测试；最终成功率、多任务和完整碰撞约束验证仍未完成。未生成 GR00T 成功视频、模型成功率或综合展示视频。GR00T 框架位于 `simulation/Isaac-GR00T`；已有适配器位于 `simulation/RoboTwin/XPolicyLab/policy/GR00T_N17`。

原数据没有逐帧显式时间戳；当前遵循官方 15Hz 转换约定。原始 `save_freq=15`、物理 250Hz 的名义间隔为 0.06 秒，还存在运动边界采样，需在闭环中核对。官方训练的 sharded dataset 工厂目前不支持内置验证集，应对独立验证集做离线 checkpoint 评估，不以训练 loss 替代验证 loss。

预计超过一小时的训练必须先获得用户确认；2000 步耗时将在短训练实测每步时间后估算，不以其他 GPU 的速度冒充 A6000 测量结果。外部 checkpoint 备份尚未配置目标。

## 100 步真实微调结果（2026-10-10）

Cosmos 与 GR00T 访问均通过。训练数据保持原 40/5/5 划分，14 维双臂配置与训练集归一化检查通过。100 个 optimizer steps 使用 micro batch 4、梯度累积 4（有效 batch 16），冻结视觉及 LLM，训练投影、动作模块及 VLLN，采用 BF16 autocast。

训练循环耗时 341.5075 秒（含 checkpoint 保存）；含模型初始化及最终保存的全过程约 8 分 20 秒（5 秒 GPU 采样精度）。排除第 50 步保存及最后保存暂停后，实测平均每步约 1.583 秒。第 10/100 步记录 Loss 1.2403/1.0069，整体平均 1.0774086。GPU 采样峰值 38956 MiB（38.04 GiB），无 OOM。

完整训练 checkpoint：`checkpoints/lift_pot_short/checkpoint-100`，含模型、optimizer、scheduler 与 trainer state。训练日志 `logs/lift-pot-short-training.log`；显存采样 `logs/lift-pot-short-gpu.csv`；汇总 `results/short_training_summary.json`；Loss CSV `results/short_training_loss.csv`；曲线 `results/training_curves/lift_pot_short.png`。

官方 Gr00tPolicy 已重载 checkpoint-100，在独立验证轨迹上完成真实推理。输出左臂 [1,40,6]、左夹爪 [1,40,1]、右臂 [1,40,6]、右夹爪 [1,40,1]，均有限；首次调用 2.9898 秒，推理峰值 torch allocated 6168.65 MiB（与训练 NVML 总显存指标不同）。记录 `evaluation/short_checkpoint_policy_check.json`。一次前向不构成验证 Loss、稳定推理延迟或闭环成功率。

正式训练脚本 `scripts/finetune_lift_pot_2000.sh` 已获授权并完成 2000 步。以下为启动前配置与预算记录：以 checkpoint-100 权重热启动，重建 optimizer 与 2000 步学习率计划，同样 batch 配置，每 500 步保存模型。模型单独保存可减少 I/O，但不含正式训练 optimizer 恢复状态；短训练完整状态保留。实测计算投影 52.8 分钟，含初始化、数据加载及保存预计 1.0–1.3 小时；按用户提供 $0.69/h 估算 $0.69–$0.90，非账单报价。配置及估算见 `configs/lift_pot_2000_estimate.json`。该轮已获确认并执行；目前用户要求不得再次训练。

复查 checkpoint 前向命令（无需重复训练）：
```bash
source scripts/project_env.sh
.venvs/gr00t/bin/python scripts/check_finetuned_policy.py --checkpoint checkpoints/lift_pot_short/checkpoint-100
```

## 2000 步模型闭环实测（2026-10-10）

正式训练已完成 2000 optimizer steps，训练循环 3339.093 秒（55分39秒），末次 Loss 0.0224、全程平均 0.09167034。最终上层目录重复导出因个人磁盘配额失败；完整 `checkpoints/lift_pot_2000/checkpoint-2000` 已实际重新加载，其模型分片、配置、processor、statistics 和本体参数均保留。本轮未再次训练、下载数据或生成专家示范。

**真实模型闭环已经运行，但未成功完成任务。** 输入为当前 SAPIEN RGB、左右关节/夹爪状态及语言；输出为官方 Gr00tPolicy 对 checkpoint-2000 的真实预测。模型在已有 `.venvs/gr00t` 环境运行，仿真在 `.venvs/robotwin`，本机桥接避免重新安装或混装依赖。无专家脚本调用，无动作回放。每次预测、执行目标和观测状态保存在 `evaluation/lift_pot_gr00t_*_actions.npz`。

完整测试：seed 200000–200004 共 5 个独立种子，0 成功；再对 seed 200002 做 16 步采样、4 次预测平均的诊断，仍失败。总计 6 个完整 rollout / 5 个独立种子 / 0 成功。各轮参数不同，不能当作单一固定策略的正式成功率；尚未进行 20 次评测。早先 TOPP 诊断被主动中止、一次场景初始化中止，均不计完整试验。

具体修复与证据：
- 官方单目标 TOPP 在诊断中执行 43 个目标推进约 18.2 秒，明显慢于原数据每帧约 60ms；改用官方 robot drives，每目标 15 个 250Hz 物理步。尝试 10 与 40 步执行块。
- 旧数据 JPEG 无 XPL-RGB1 标记，官方 codec 的 cv2 解码结果已经是 RGB。此前转换中的额外 BGR2RGB 使已训练数据红蓝通道相反。训练 AV1 首帧与交换后源图 MAE 0.955，与正确源 RGB MAE 2.890。在线输入加入通道匹配，原视频仍为正常 RGB；原数据和权重未改写。转换脚本已修正，未来新数据使用官方 codec；现有模型仍须启用在线匹配。见 `evaluation/image_channel_audit.json`。
- 纯 CPU URDF FK 对原数据每条 3 帧、每臂 150 个样本检查，左右位置误差中位数 2.68mm/1.51mm、最大约 1cm，支持单位、关节顺序及资产运动学兼容。见 `evaluation/dataset_kinematics_check.json`，不涉及动作回放。
- 通道匹配试验 seed 200003，在第 80 步关节跟踪最大误差约 0.0124rad，但左右 TCP 距指定抓取点仍约 0.228m/0.259m，未抓稳。后期跟踪误差增大并推歪锅；seed 200004 的锅被推落桌面。重复 seed 200002 的 16步/4样本诊断最后左右距离约 0.313m/0.0667m。由此不能用控制速度或权重加载问题解释全部失败，策略抓取定位仍需进一步分析。

保存视频：每个完整原始试验 `videos/lift_pot_gr00t_*_raw.mp4`，均为 960×720、31.25fps、12.064秒，ffprobe 与 FFmpeg 全片解码通过。代表性标注失败视频 `videos/lift_pot_gr00t_failure_seed200002.mp4`，960×720、30fps、12.066667秒，明确 FAILED，全片解码通过。seed 200002 曾将锅抬到约 0.836m，但锅倾斜、官方成功判定未通过，不能标为成功。**未生成 `videos/01_lift_pot_gr00t.mp4`。** 结果见 `results/gr00t_closed_loop_summary.json`、`results/metrics.csv` 与 `evaluation/gr00t_video_checks.json`。

磁盘：只清理 uv 可重建安装缓存，工具报告移除 88882文件/21GiB，因硬链接共享，项目实际占用下降约 9GiB。清理后项目约 175.46GB（163.42GiB），1MiB 写入+fsync 和全部录像写入均通过。df 显示共享存储余量，不能代表个人配额，容器无可用配额查询工具，精确个人剩余量未获验证。所有训练 checkpoint、数据、代码、视频均保留；直接使用现有 checkpoint，不再导出模型副本。

复现通道匹配闭环（两个已有环境，两个终端；使用新 tag/seed，避免覆盖已存视频）：
```bash
cd /workspace/GR00T_DualArm_Project
source scripts/project_env.sh
# 终端 1；等待日志显示 ready，再运行终端 2：
.venvs/gr00t/bin/python -u scripts/serve_gr00t_local.py --denoising-steps 16 --ensemble 4
# 终端 2：
.venvs/robotwin/bin/python -u scripts/evaluate_lift_pot_gr00t.py \
  --start-seed 200005 --episodes 1 --max-actions 200 --execute-horizon 10 \
  --control fixed --match-legacy-training-channels --tag next_diagnostic
```

下一步无需训练的诊断应量化独立验证集 40 步动作的关节与 FK 末端误差，再对比实测关节反馈、较短执行块和目标速度限制，区分定位偏差、轨迹多解及后期碰撞/跟踪误差。当前不会增加训练步数，也不会以专家动作替代失败模型结果。所有本轮推理和仿真进程已停止。

本轮结束 `/workspace` 实测总占用 176.61GB；若用户给定配额确为十进制 200GB，估算余量约 23.39GB（21.78GiB），这不是个人配额实查结果。


## 最新恢复诊断结论（2026-10-10）

发现 40 步 sharded sampler 排除了 1560 个真实训练观测，所有 990 个闭合夹爪观测均未采样。数据文件的 4444 帧不等于模型实际用到的 4444 个观测。末段专家状态上的夹爪预测 MAE 约 0.51；这比单纯“训练步数不足”更直接地解释闭环失败。详见 [根因报告](results/gr00t_root_cause_analysis.md)、[版本审计](results/framework_version_audit.md)、[恢复试验](results/gr00t_recovery_test.md)。本轮 9 个完整模型 rollout 全失败，训练场景排除在正式成功率外；尚无成功命名视频。

已准备正确 RGB + 完整真实观测覆盖的独立 v23 数据版本，官方 sampler 检查为 4444 个真实训练起点、990 个闭合状态，人工终点保持目标不作为输入观测。新训练统计已刷新，旧权重/数据保留。建议 B 对照从官方 N1.7 基础权重训练；[配置和预算](configs/lift_pot_rgb_B_proposal.json) 已准备，预计超过一小时，尚未获本轮确认且未启动。

可复现数据检查：
```bash
source scripts/project_env.sh
.venvs/robotwin/bin/python scripts/audit_training_coverage.py
.venvs/gr00t/bin/python scripts/check_terminal_pad_sampler.py
```

修复后服务器设置实际去噪步数于 `policy.model.action_head`；旧脚本对外层 model 的设置不生效，历史标称 16 步结果实际为默认 4 步。评测脚本未显式给定采样参数时继续使用服务器默认值。原有 `.venvs` 无需重新创建；容器重建后若渲染器创建失败，检查 EGL/GL 系统用户库。
