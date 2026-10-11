"""Write concise diagnosis from existing measured artifacts; no simulation or training."""
import json,csv
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=json.loads((root/'evaluation/B_test20_diagnostic_summary.json').read_text());rgb=json.loads((root/'evaluation/B_rgb_pixel_audit.json').read_text());rows=list(csv.DictReader((root/'results/B_test20_grasp_distances.csv').open()))
f=lambda v:f'{float(v)*100:.2f}'
lines=['# Lift Pot：B 模型 20 次独立测试诊断','',
'2026-10-10。仅读取已有结果并做 CPU 数据分析，未重复启动仿真、未训练、未覆盖 checkpoint。B 为官方 GR00T N1.7 基础权重 + 正确 RGB/末段采样修复的 2000 步模型。固定配置：预测 40 步、执行前 5 步、去噪 4 步、每动作 15×4 ms，约每 300 ms 重规划。', '',
'**结论：20 次实际执行，0 成功、20 失败，0 次初始化或推理中断；训练场景 seed 1 真实模型成功，但不计正式成功率。动作目标下发和当前 RGB 没有发现错误。失败表现更符合抓取接触偏差、实际关节受接触约束后 command-state 反馈失真，以及闭环状态偏离示范的共同作用。无法用现有日志给这些因素分配严格因果占比，也不能断言再训练即可解决。**','',
'## 1. 逐种子抓取距离','',
'下面统一使用每次运行“最后一次推理前观测”的 TCP 到当时锅抓取点距离，单位 cm；不是预测关节动作的 MAE。重规划每 5 步一次，正常运行最后记录在第 195 步，结束为第 200 步。200005、200013 提前掉落中止，没有完整末帧 TCP/接触点，不能虚构其终点值。完整末帧 18 次另行统计。','',
'| 种子 | 锅模型 | 最后观测步 | 左距离 cm | 右距离 cm | 两臂最佳同帧最大距离 cm |','|---|---|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['seed']} | base{r['model_id']} | {r['last_observed_action_count']} | {f(r['last_observed_left_m'])} | {f(r['last_observed_right_m'])} | {f(r['best_simultaneous_max_m'])} |")
lines+=['','| 统计口径 | 左均值 / 最小 / 最大 cm | 右均值 / 最小 / 最大 cm |','|---|---|---|']
for key,label in [('last_observed_distance_all20','20 次最后观测'),('final_distance_available','18 次完整终点'),('minimum_observed_distance_all20','20 次各臂观测最小距离')]:
 a,b=s[key]['left'],s[key]['right'];lines.append(f"| {label} | {' / '.join(f(a[k]) for k in ['mean','min','max'])} | {' / '.join(f(b[k]) for k in ['mean','min','max'])} |")
lines+=['','“约 12 cm”是部分种子/时刻，不能代表平均值。掉落及后期发散会放大终点距离。各臂最小值可能来自不同帧，不能拼成双臂成功；最佳同帧 max(L,R) 为 3.76–10.48 cm。官方在每个物理步检查，20 次均未满足高度 >0.82 m、左右距离各 <3 cm、锅朝上 dot >0.8。完整数据见 [距离 CSV](B_test20_grasp_distances.csv)。','',
'## 2. 动作接口与反归一化','',
f"逐动作核对现有 NPZ：GR00T 已解码输出的前 5 步，与真实记录的 RoboTwin drive target 最大绝对差为 **{s['model_target_to_executed_drive_max_abs']:.1f}**（20 次全部）。顺序为左 6 关节、左夹爪、右 6 关节、右夹爪；关节单位 rad；左右关节按官方 URDF 顺序。模型 processor 内使用相对关节表示，`decode_action` 已反归一化并转为绝对目标，控制端不再加当前状态或二次反归一化。B processor 实测关节归一化往返误差 ≤1.11e-16；初始夹爪 1 的 0.03769 截断来自 q99=0.96231，不能解释后期 10 cm 以上空间误差。",'',
'控制端直接给关节 drive，无额外世界/机器人坐标变换；成功判定使用 RoboTwin 原生世界 TCP 和物体接触点。FK 用同一 URDF、机器人初始 pose、global/delta 旋转及 gripper_bias，故未发现动作坐标系转换错误。既有原始专家回放通过同一 60 ms 接口成功，但只验证了 seed 1，不能据此保证新场景接触执行。','',
'## 3. 同一初始状态的专家与模型差异','',
'对训练场景 seed 1，原始专家 q[0] 与已录制 B 在线输入关节逐项相同（最大差 0）；此前场景审计确认相同种子相机几何和初始锅位置。直接比较该在线首块预测与原始专家 q[1:41]，不运行任何新专家或模型任务。渲染/JPEG/H264 有压缩像素差，不能称像素逐位相同。','',
'| 预测步 | 左/右关节 MAE rad | 左/右 TCP 误差 mm |','|---:|---|---|']
for r in s['live_seed1_same_initial_expert_comparison']:lines.append(f"| {r['horizon']} | {r['left_joint_mae_rad']:.5f} / {r['right_joint_mae_rad']:.5f} | {r['left_tcp_error_m']*1000:.2f} / {r['right_tcp_error_m']*1000:.2f} |")
lines+=['','该评测每块只执行前 5 步，第 40 步误差不能当成已经下发的误差。另有 40 个训练、5 个验证 episode 相同观测的离线专家比较，初始第 5 步验证集 TCP 平均误差左 5.47 mm、右 8.33 mm；第 40 步为 28.85/32.89 mm。独立测试 20 个种子没有配套专家轨迹，因此**尚无这 20 个完全同场景专家与模型动作的直接对照**，不能用相近训练轨迹替代。','',
'## 4. 误差从哪里来','',
'记录中的 drive target 是已下发命令，不代表机械臂实际到达。当前输入仍是 drive target；实际关节不能跟随时，模型会以“已到达目标”的状态继续规划。下表为 URDF FK(command) 与 FK(real qpos) 的 TCP 跟踪差，按推理观测统计，属于执行偏差，不是独立的模型预测误差。','',
'| 已执行动作区间 | 左 TCP 跟踪差均值 mm | 右 TCP 跟踪差均值 mm |','|---|---:|---:|']
for phase,r in s['phase_tracking'].items():lines.append(f"| {phase} | {r['left_tcp_tracking_m']['mean']*1000:.2f} | {r['right_tcp_tracking_m']['mean']*1000:.2f} |")
lines+=['',
'早期约 1.5 mm，接触后显著增大，150 步后约 125/172 mm；这修正了“部分早期跟踪误差小，所以执行无问题”的推断。模型抓取偏差可能先触发碰撞/滑移，再由错误的 command-state 反馈放大；仅凭日志无法证明先后因果。现有目标名义速度最大约 2.18 rad/s，与专家最大约 2.22 rad/s 接近，单纯限速 2 rad/s 很可能不够。反例 seed 200004 在最后观测第 195 步右抓取点距离 63.26 mm，但 drive-real TCP 差只有 2.48 mm；根据三角不等式，当前命令 TCP 到抓取点仍至少差 60.78 mm。因此即使完美执行当前命令，也不能使这一帧右臂进入 30 mm 阈值，模型/物体相对抓取定位偏差不能被忽略。seed 200007 则出现大幅跟踪发散，seed 200013 的终点距离主要受到物体掉落影响；不能将所有终点误差都叫作关节预测误差。动作间隔为 60 ms，与原始 save_freq=15/250 Hz 名义一致；原始边界帧缺精确时间戳，尚不能彻底排除时间对齐影响。5 步开放执行也可能错过接触修正。','',
'## 5. RGB 与训练/测试分布','',
f"本次直接复查 B 实际使用的 v23 数据：50 条 episode ×3 相机的首帧共 {rgb['sample_frames']} 帧，{rgb['rgb_better_count']} 帧全部支持正确 RGB。相对原始官方解码 RGB 的像素 MAE {rgb['mean_rgb_mae']:.3f}/255，错换通道则 {rgb['mean_swapped_mae']:.3f}/255。三视角 uint8 HWC 240×320；在线进入模型为 [1,1,240,320,3]。20 次元数据均 `model_input_red_blue_swapped=false`，使用 B 自己的 processor；A 才需要历史交换补偿。没有证据需要给 B 再换红蓝通道。抽检覆盖首帧，未逐帧像素审计；代码转换使用同一官方 marker-aware 解码。",'',
'重要分布差异：原始 50 条示范全部是 base0，训练/验证/数据测试分别 40/5/5，没有 base1；独立仿真为 base0 12 次、base1 8 次，二者均 0 成功。base1 是物体模型分布外，不能用 base0 验证集低误差证明能泛化到它；但 base0 也 0/12，故模型型号差异不是唯一原因。训练场景只验证了 seed 1 成功，并未测出全部训练场景成功率。离线 train/validation 近端误差都低，不能简单判为“明显参数过拟合”；已证实的是闭环泛化缺口与接触后状态分布偏离。','',
'## 6. 优先处理与是否再训练','',
'不建议重新做相同的正确 RGB 微调：B 已完成该对照，且学会了闭合夹爪。现有 B 在训练场景已成功，说明模型、双臂输出、控制接口具备完成任务的能力；新场景能否仅靠执行修复成功尚未验证。下一步优先非训练、逐因素诊断：先比较真实关节状态与 drive-state 输入（保留原归一化和时序）；再单独缩短执行块 5→1；再检查实际 qpos 起点插值和接触阶段跟踪约束。真实 qpos 作为状态与原训练 drive-state 有语义差别，不能盲目混用；夹爪当前 real getter 仍返回命令值，若需要物理夹爪反馈须单独测量。之前 A 的实际状态试验失败不能代替 B 的验证。本次未启动上述新测试，不改已有 20 次正式结果。','',
'只有非训练修复仍失败且确认覆盖不足时，才考虑 C：收集两个锅模型及不同抓取/接触恢复状态的高质量示范，保持正确 RGB 和末段覆盖，从官方 N1.7 基础权重另存训练。参考本轮 2000 步 45.3 分钟，纯训练按历史假设 $0.69/h 约 $0.52；含加载、保存的预算仍可按 1–1.3 小时、$0.69–0.90，**当前账单费率未核实，示范采集/评测时间不包含在内**。预留约 50 GB 容器盘及约 13 GB 新权重持久存储，个人持久卷剩余额度仍未确认。A/B 和现有数据全部保留；不执行 C，任何预计超过 1 小时的新 GPU 训练仍须单独确认。','',
'## 曲线与代表性原始失败视频','',
'![20 种子左右抓取距离与锅高度](B_test20_contact_error_curves.png)','',
'![同初始专家预测误差](B_same_initial_expert_error_curves.png)','',
'![代表性接触误差与关节跟踪](B_representative_tracking_curves.png)','',
'[200004：锅高度 0.874 m，但右侧末帧距离 6.21 cm](../videos/lift_pot_gr00t_recovery_B_h5_d4_test20_seed200004_raw.mp4)；[200007：后期跟踪发散](../videos/lift_pot_gr00t_recovery_B_h5_d4_test20_seed200007_raw.mp4)；[200013：未见锅模型 base1，掉落中止](../videos/lift_pot_gr00t_recovery_B_h5_d4_test20_seed200013_raw.mp4)。这些均为完整原始模型仿真，非专家回放。','',
'复现只读诊断：`source scripts/project_env.sh`；`.venvs/robotwin/bin/python scripts/extract_B_same_initial.py`；`.venvs/robotwin/bin/python scripts/audit_B_rgb_pixels.py`；`.venvs/gr00t/bin/python scripts/diagnose_B_test20.py`；`python scripts/write_B_test20_diagnosis.py`。分析 JSON：`evaluation/B_test20_diagnostic_summary.json`；每帧跟踪：`results/B_test20_joint_tracking.csv`。']
(root/'results/lift_pot_B_test20_diagnosis.md').write_text('\n'.join(lines)+'\n')
print('Wrote results/lift_pot_B_test20_diagnosis.md')
