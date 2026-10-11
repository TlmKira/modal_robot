import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'results/C_preparation';summaries=[json.loads(p.read_text()) for p in sorted(out.glob('pilot*_quality.json'))];records=[r for s in summaries for r in s['records']];accepted=[r for r in records if r['accepted']];hold=[json.loads(p.read_text()) for p in sorted((root/'evaluation/B_saved_action_hold').glob('*.json'))];manifest=json.loads((out/'protected_artifacts_manifest.json').read_text());size=sum(r.get('bytes',0) for r in records);mean=sum(r['bytes'] for r in accepted)/max(len(accepted),1)
text=f'''# B 成果保护与 C 数据扩充可行性实测

## 权重与展示

B checkpoint 全部 {sum('matches_prior' in r for r in manifest)} 个文件重新流式 SHA256 校验，与此前持久复制记录一致：{all(r.get('matches_prior',True) for r in manifest)}。现有两段模型成功原视频、历史 results/evaluation 报告与动作日志登记，共 {len(manifest)} 项。没有修改 checkpoint、删除旧数据或升级框架。校验清单不是异地备份；现有原始持久卷权重及既有临时备份均保留。

首选展示候选为 [seed19 原始模型视频](../../videos/B_contact_feedback_qpos_full_h5_seed19_raw.mp4)，标注：GR00T N1.7 微调模型、RoboTwin 仿真、真实模型执行、官方成功判定通过。原 15 次受控诊断与 0/20 独立测试结论不变；seed19/107 已用于调参，不再列入新盲测。

## 独立保持测试

使用保存的真实 B 模型执行指令重建两场成功，然后冻结最后 drive target、关节速度设零，继续 500 个 4ms 物理步。此测试明确是保存模型指令回放，无新在线推理，不是专家回放，也不加入新模型成功率。

|种子|重现首次官方成功|2秒末官方条件|2秒末锅高度(m)|
|---|---|---|---|
'''
for d in hold:text+=f"|{d['seed']}|{d['replay_reproduced_official_success']}|{d['hold_final_success']}|{d['telemetry'][-1]['pot_position'][2]:.6f}|\n"
text+='''
两场都无法全程维持官方成功条件。首次成功时高度只略越过 0.82m，冻结目标导致高度回落；不能由此直接断言物体完全滑落。需要后续在线继续预测以及持续双侧接触/姿态日志，当前没有证明模型可稳定保持。原首次成功判定保留。曲线：[锅高度](B_saved_action_hold_height.png)，完整日志 evaluation/B_saved_action_hold/。

## 官方专家小规模新数据

'''
text+=f'实际尝试 {len(records)} 个全新种子；通过官方成功、规划、关节路径、有限值、夹爪命令范围、RGB 类型、必需三相机、严格时间戳等基础检查 {len(accepted)} 条。原数据完全保留，新源数据独立 datasets/lift_pot_C_pilot01/ 与 pilot02/。\n\n'
text+='|型号|尝试|基础检查通过|\n|---|---|---|\n'
for i in [0,1]:text+=f"|base{i}|{sum(r.get('model_id')==i for r in records)}|{sum(r.get('model_id')==i and r['accepted'] for r in records)}|\n"
text+='''
通过名单以独立 pilot*_quality.json 为准。首轮生成器错误要求相机数量恰为三，实际官方提供额外 front 相机；已修为必需三相机子集，并独立复核首轮，源文件和原始 manifest 不改写，因此一个原名 rejected 文件可在复核后通过基础检查。它尚未发布为训练轨迹，不能仅凭文件名选样。

新原始图像为官方 get_rgb 的 Color 前三通道、uint8 HWC，直接无损 HDF5 保存，无 OpenCV RGB/BGR 转换。三训练相机均 240×320；额外 front 单独保留。未来视频编码后须逐相机抽检 RGB 误差。状态为绝对 drive targets 14维；真实 arm qpos 与实际 base/mimic finger qpos 米为独立字段。原 legacy real jointState 中夹爪仍是命令，字段明确标注，不冒充真实夹爪。

按真实 scene.step 每15步采样，保存物理tick和实际timestep时间戳；末尾短间隔可见，不应按60ms标签强行训练。训练导出需排除离网格尾帧、构造未来 action 并通过同接口专家回放。前两轮仅完成基础质量检查，没有完整逐tick接触、碰撞及保持验收，现阶段通过名单是候选，不是最终高质量训练发布。生成器已增加后续采集的采样接触和左右TCP距离字段；采样接触仍不能替代逐tick连续持续时间。

官方 Lift Pot play_once 只有对齐、闭合、抬升，没有经过验证的偏差恢复分支。base1 多次出现规划失败或 target_pose None；保留失败日志并隔离，不能作为监督成功示范。不扩大 base1 批量采集，不引入无标签模型失败轨迹。

## 空间、费用与下一步

'''
text+=f'两轮原始记录合计 {size/1e9:.3f}GB；成功候选平均 {mean/1e6:.2f}MB，成功候选原始数据50/100条粗估 {mean*50/1e9:.2f}/{mean*100/1e9:.2f}GB（当前小样本，仅估算）。另预留失败源数据、转换副本、视频、统计缓存，建议数据阶段保守预算15–25GB。新 checkpoint 保存与根导出各约12GB，训练及中间产物另预留30–60GB；不复制官方基础权重。容器临时盘615GB剩余，网络盘共享236TB剩余不代表个人配额，当前个人配额与GPU单价未知，已向用户询问。\n\n'
text+='''下一步先扩充可靠 base0、验证接触/抬升/保持与控制回放，冻结新训练/验证/盲测划分，再发布正确RGB、等间隔 drive-state 数据。第一批目标50–100条，通过质量和预算检查后才扩大；尚未完成50条或C训练。暂不直接启动200–300条大批采集。

[C训练设计](C_training_design.md) 已记录官方基础权重与B继续训练的取舍、100–300步短训检查、2000步正式预算、统一固定控制与新盲测。当前没有新增训练、付费资源或长时任务；正式训练等待配置、时间、费用、空间完整报告后用户确认。
'''
(out/'C_preparation_feasibility.md').write_text(text)
