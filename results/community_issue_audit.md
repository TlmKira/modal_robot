# 社区 Issue 核实（2026-10-11，CPU only）

核查方式：读取官方 GitHub Issue 正文、评论和本项目固定源码。API 原始快照保存于 `results/community_audit_sources/`。没有GPU、模型加载、图形仿真或框架更新。社区描述不等于维护者确认。

本项目 RoboTwin commit `ea8b21121ebb3cd201ff5b3fe361944ac94eda3f`；Isaac-GR00T commit `d2b7e75b937e3ec9aa5dbc798f08b89692c49734`。模型始终 GR00T N1.7。下面日期来自 GitHub API。

## RoboTwin #489

[Joint state observations use drive targets instead of actual qpos](https://github.com/RoboTwin-Platform/RoboTwin/issues/489)，2026-08-20创建，核查时 Open，API返回没有评论。作者指出关节观测来自drive target，并询问这是否有意设计；没有给出可据以锁定适用版本的commit，也没有维护者确认或修复证据。

**本地适用性由代码独立确认**：`envs/robot/robot.py:494–524` 的普通关节读取是drive target，real读取才读实体qpos；但real函数的夹爪元素仍是命令。`envs/_base_task.py:488` 的get_obs采用普通读取。本项目BC runner也显式采用drive。RGB反映实际位姿，而关节状态反映目标；接触后可不一致。

B/C训练同样使用drive-state，所以这里有确定的状态语义/物理反馈局限，尚不能称为本项目“训练推理状态映射错误”。直接改为qpos会改变模型输入与relative动作解码参考。旧B受控实验显示反馈改善跟踪但未提高3种子总体成功，不能宣布qpos替换是已验证解法。

## RoboTwin #482

[Invalid grasp contact point makes 001_bottle variant 19 ungraspable by ALOHA](https://github.com/RoboTwin-Platform/RoboTwin/issues/482)，2026-08-07创建，Open，API返回没有评论。正文显式报告版本`c659e0dc7eb62cb9d2a4e112d2285dc659f1a017`，涉及pick_diverse_bottles、001_bottle variant19；作者给出几何分析和下游验证，尚无维护者确认上游修复。

**瓶子资产结论不适用于Lift Pot锅模型**，不能据此认定base0/base1锅有相同抓取点Bug。相关的是正文指出的数据收集/评测expert可行性筛选。该筛选也已从我们固定版本的`eval_policy_xpolicylab.py:787–835`核实：默认expert_check=True，先专家play_once，失败seed被跳过，再同seed重新初始化供模型评测。

昨日BC种子只按base0类型预选，没有expert_check；旧20独立测试同样没有此门控且包含base1。因此：

- 同一组20base0种子、同控制的B/C比较可用，结果0/20与9/20不变。
- 它们是“未expert过滤的base0固定控制评测”，不能作为官方默认专家可行分布上的直接可比基准；官方还可能使用不同时间、episode上限及动作执行器。
- 不能在看到模型失败后删除专家不可行seed并重算为新的盲测成功率。以后可在独立新候选种子上先冻结expert通过集合，再测试两模型；同时报告raw及expert-conditioned两种口径。

## RoboTwin #202

[Arms Don't Reset to Origin Post Success During Eval](https://github.com/RoboTwin-Platform/RoboTwin/issues/202)，2025-09-17创建，Open。报告custom ACT/eval.sh成功后次episode继承末态，涉及lift_pot；没有明确commit或可靠复现步骤。维护者表示没有遇到并请求视频。**这是未确认报告，不能当已证实框架Bug。**

我们的正式runner每episode新建lift_pot对象、setup_demo新场景/机器人、初始化计数/夹爪，结束close_env；每seed重置模型随机种子。保存的20对B/C初始锅矩阵、drive14维与真实14维逐项差为0；C成功后的下一episode初始drive也恢复相同home。没有日志证据支持本项目遭遇该问题。real qvel、夹爪mimic细项及控制器内部全状态没有完整初始快照，因此不能CPU证明所有隐藏状态完全reset。建议增加初始不变量和异常finally清理，不先修改官方reset。

## NVIDIA Isaac-GR00T #414

[The success rate is very low on real robot even though the open loop testing looks good](https://github.com/NVIDIA/Isaac-GR00T/issues/414)，2025-10-27创建，Open。报告Unitree G1 pick/place，约600条示范、100000步，旧`scripts/gr00t_finetune.py`与BaseDataConfig/ComposedModalityTransform接口；只配置左臂/左手及两相机。正文未明确精确模型版本/commit，不能从标题认定为N1.7。两条评论询问真实推理视频及数据采集，未给出经确认根因/修复。

可类比的是开环误差好不保证闭环成功：本项目B保存的训练grasp一步TCP误差约毫米，正式闭环仍0/20。离线误差对专家目标，不衡量接触物理、夹爪与分布外恢复。C完整分阶段离线验证尚未计算，不能声称C已经证明“离线好而闭环差”。#414不提供本项目该升级、换模型或加训练步数的证据。

## 核实结论

四个Issue均没有可确认的上游修复可直接应用。#489的目标/实测分离、#482的expert筛选在本地可验证；#202没有本项目复现证据；#414仅是跨旧接口的症状类比。继续冻结现有框架与checkpoint，先做接口观测和控制消融。
