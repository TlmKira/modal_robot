# GR00T N1.7 root cause analysis — recovery 2026-10-10

**主要根因：40 步预测窗口将每条轨迹最后 39 个观测排除出训练，990 个夹爪闭合观测全部未被采样。末段专家状态上的夹爪预测误差约 0.51；低训练 Loss 和早期窗口低 MAE 因此不能证明会完成抓取。正确 RGB 是另一项确定的数据缺陷。**

## Verified findings

1. Container recreation removed EGL/GL user libraries. Python virtualenvs, CUDA and checkpoint files survived. Restoring the user libraries restored renderer creation. This blocked new simulation runs, but does not explain historical completed policy failures.
2. Original expert episode 1 replayed through the exact shared `recovery_control.execute_fixed` used by model evaluation, at 15 physics ticks (60ms) per target. Official success: True; 105 targets, left/right contact distances 0.005463/0.002450m. This is an expert diagnostic, excluded from model results.
3. Checkpoint index/header validates 1030 tensors and all shard offsets; real Gr00tPolicy reload and inference completed. Config/processor/statistics/embodiment metadata are present. Formal checkpoint is model-only and does not contain formal optimizer/scheduler state; this does not block inference. No full external backup exists.
4. All 50 source episodes × 3 cameras were compared at the first frame. All 150 comparisons favor red/blue-swapped source RGB versus training AV1. Average correct-RGB MAE 4.4456; swapped MAE 0.9011 pixel levels. Source is uint8 240×320×3. This proves broad episode/camera impact; it is not an exhaustive audit of all video frames. Legacy JPEGs have no XPL-RGB1 marker, so the official codec returns RGB directly. Extra cv2 BGR2RGB in the historical project conversion caused the swap. Current conversion script is fixed; old datasets and weights remain untouched.
5. Existing policy gets raw uint8 HWC video [B,T,H,W,C]. Live legacy matching swaps exactly once before the same saved official processor. Model and processor are in eval mode; no second normalization or relative-action addition belongs in the adapter. Saved config uses relative NON_EEF joints, absolute grippers, percentile/min-max scaling, action horizon 40. Short execution prefixes keep that horizon/statistics unchanged.

## Offline action and TCP errors

Primary CSV combines 225 full-window calls and 90 late-stage calls (1395 horizon rows). Original full-window CSV is preserved as `offline_action_metrics_full_chunks.csv`. The table below is the full-window subset: 225 real-model calls, 5 evenly spaced valid 40-step anchors for each of 40 training and 5 validation episodes. Seed 20261010, 4 denoising steps, no ensembles. Tail anchors with insufficient future targets are excluded. Metrics are sampled, not exhaustive per-frame evaluation; orientation is geodesic radians; TCP FK includes RoboTwin global/delta rotation and 0.12m gripper bias. Predictions are compared to expert commanded targets, not measured physics. Raw predictions/targets are preserved.

| Split | Step | Samples | L/R joint MAE rad | L/R TCP m | L/R orientation rad | L/R gripper MAE |
|---|---:|---:|---|---|---|---|
| train | 1 | 200 | 0.00191/0.00183 | 0.00112/0.00104 | 0.00439/0.00402 | 0.01865/0.01648 |
| train | 5 | 200 | 0.00800/0.00752 | 0.00478/0.00490 | 0.02042/0.01729 | 0.01714/0.01543 |
| train | 10 | 200 | 0.01477/0.01428 | 0.00932/0.00940 | 0.03716/0.03347 | 0.01564/0.01547 |
| train | 20 | 200 | 0.02677/0.02468 | 0.01699/0.01593 | 0.06033/0.05596 | 0.01065/0.01062 |
| train | 40 | 200 | 0.03457/0.03600 | 0.02530/0.02411 | 0.07669/0.06910 | 0.00984/0.00988 |
| validation | 1 | 25 | 0.00371/0.00430 | 0.00154/0.00166 | 0.00759/0.00643 | 0.01858/0.01763 |
| validation | 5 | 25 | 0.01621/0.01881 | 0.00816/0.00889 | 0.03962/0.03115 | 0.01639/0.01681 |
| validation | 10 | 25 | 0.02931/0.03076 | 0.01734/0.01488 | 0.06550/0.05192 | 0.01820/0.01929 |
| validation | 20 | 25 | 0.06868/0.05491 | 0.02102/0.02469 | 0.09479/0.08970 | 0.01266/0.01261 |
| validation | 40 | 25 | 0.11991/0.09151 | 0.03544/0.02978 | 0.14678/0.09714 | 0.01314/0.01326 |

![Offline action errors](offline_action_error_curves.png)

The low first-step error on both splits argues against gross joint-order/unit/decoder failures. Validation long-horizon errors exceed training errors. This supports short-prefix testing, but is not evidence of task success or proof that channel matching fixes distribution shift. Offline inputs are expert states; accumulated closed-loop errors and unfamiliar camera/object configurations are absent.

## Saved processor round-trip

Measured raw→normalized→decoded action max errors: `{"left_arm": 1.1102230246251565e-16, "left_gripper": 0.03768843412399292, "right_arm": 1.1102230246251565e-16, "right_gripper": 0.03768843412399292}`. Clipping of training percentile outliers can make a round-trip lossy; do not “fix” this by applying a second denormalization.

## Same-seed source/live geometry

Original episode 1 compared with current clean scene seed 1.
- head_camera: source/live shapes [240, 320, 3]/[240, 320, 3]; intrinsic max difference 0.000000, extrinsic max difference 0.000000.
- left_camera: source/live shapes [240, 320, 3]/[240, 320, 3]; intrinsic max difference 0.000000, extrinsic max difference 0.000000.
- right_camera: source/live shapes [240, 320, 3]/[240, 320, 3]; intrinsic max difference 0.000000, extrinsic max difference 0.000000.

Raw state, matrices and screenshot pairs are preserved in `evaluation/live_source_scene_audit.json` and `results/screenshots/recovery_*`. Seed equality alone cannot establish identical scenes across framework/data versions.

## Inference configuration defect

Historical server wrote `policy.model.num_inference_timesteps`, an unused outer-model attribute. Actual sampler lives in `policy.model.action_head.num_inference_timesteps`; checkpoint default is 4. Historical nominal 16-step tests therefore remained 4-step. Corrected setter now targets action_head. The interrupted `recovery_h5_d16_train` trial is explicitly invalidated and excluded from completed rollouts; its predictions/video remain preserved. This does not invalidate baseline 4-step offline metrics.

## Causal conclusions and retraining decision

Most direct failure mechanism: missing late-stage conditioning states and poor gripper closure/hold predictions, supported by the coverage audit and late-phase inference. This is a sampler/data coverage defect, not a claim that more optimization steps alone will solve the problem. Historical TOPP timing mismatch and RGB conversion error are also established defects. Current 60ms interface succeeds on original expert actions. Offline first-step model outputs are accurate on expert observations. Therefore “2000 steps is insufficient” is not supported as the sole cause. Same-seed scene/camera alignment and expert interface replay passed. Short-prefix, actual feedback, velocity constraint and actual 16-step denoising trials did not establish a successful policy. A cannot acquire untrained late-stage behavior simply by changing denormalization. RGB swap impact alone does not prove that a new model is necessary: matching the old training channels restores only channel consistency, not any other domain shift.

New terminal-padded correct-RGB v23 datasets are prepared to expose all 4444 REAL observation anchors, including all 990 closed-gripper states; 39 terminal HOLD targets per episode supply future horizon targets but are never selected as conditioning observations. New training statistics are regenerated; validation/test reuse those training statistics. No baseline framework or model is changed. New true-RGB datasets are also prepared separately as `lift_pot_{train,validation,test}_rgb_v22`, directly from original JPEGs with official decoding, preserving original split/actions/statistics; provenance in `evaluation/correct_rgb_dataset_manifest.json`. Old datasets are untouched. No new training started. B/C models do not exist. Recommendation: retrain B from official N1.7 base on correct RGB with restored REAL late-stage conditioning anchors, preserving A. This addresses missing supervision rather than simply adding steps to A. Configuration and runtime/storage/backup/cost proposal are concrete in `configs/lift_pot_rgb_B_proposal.json`; the script is prepared but not run. Obtain user confirmation before the estimated 1.0–1.3 hour run. A successful B rollout and 20 independent tests remain required; new data readiness is not model success. No invented GPU estimate or success claim.

## Late grasp/lift observations

Supplementary 90 real-model calls at 75% of each trajectory and five frames before its end, with only valid future horizons; separate `results/offline_action_metrics_late_phase.csv`. These anchors cover observation states excluded by full 40-step-window sampling. Phase labels are fractional trajectory anchors, not manual semantic event annotations.
- train grasp: first-step left/right TCP MAE 0.00167/0.00137m; gripper MAE 0.32883/0.33676.
- train lift: first-step left/right TCP MAE 0.01183/0.01070m; gripper MAE 0.51224/0.51942.
- validation grasp: first-step left/right TCP MAE 0.00141/0.00132m; gripper MAE 0.32231/0.33554.
- validation lift: first-step left/right TCP MAE 0.01185/0.01134m; gripper MAE 0.50488/0.51755.

## Direct training coverage evidence

Official sampler computes `length - horizon + 1`; recorded `allow_padding=false`. The training log has exactly 2884 anchors = 4444 - 40×39. Excluded anchors: 1560 (35.1%). Real closed-gripper observation states: 990; sampled closed-gripper states: 0. Both the sampled state and first action gripper minima are 0.5. Later future-action slots can still contain closure; this is specifically missing conditioning/near-term supervision, not absence of closure everywhere in the action chunks.

The model predicts a roughly half-open gripper even from held-out expert closed-gripper/lift states (late-phase MAE about 0.51). This explains why very small early joint MAE and low training loss coexist with failed receding-horizon grasps. Latest official main has the same sampler; upgrading alone will not restore omitted observations. `allow_padding=True` by itself also leaves the effective-length function unchanged. [Official sampler](https://github.com/NVIDIA/Isaac-GR00T/blob/main/gr00t/data/dataset/sharded_single_step_dataset.py).

The v23 training-only terminal target padding provides valid 40-step future windows for all REAL anchors without modifying the baseline framework or manufacturing evaluation actions. Exact sampler union and closed-state coverage are independently checked in `evaluation/terminal_pad_sampler_check.json`.

## Actual 16-step denoising comparison

This uses the corrected action_head setter, the same 225 anchors and RNG seed, and separate `_d16` outputs.
- train step 1: left/right TCP errors 0.00115/0.00120m.
- train step 5: left/right TCP errors 0.00506/0.00525m.
- train step 10: left/right TCP errors 0.00895/0.00956m.
- train step 20: left/right TCP errors 0.01552/0.01622m.
- train step 40: left/right TCP errors 0.02258/0.02550m.
- validation step 1: left/right TCP errors 0.00145/0.00152m.
- validation step 5: left/right TCP errors 0.00755/0.00816m.
- validation step 10: left/right TCP errors 0.01507/0.01515m.
- validation step 20: left/right TCP errors 0.02085/0.02312m.
- validation step 40: left/right TCP errors 0.03698/0.03210m.

## Corrected dataset verification

All 50 new episodes pass official LeRobotEpisodeLoader, matching original action parquet hashes and statistics. New 150 first-frame RGB pairs: correct-source MAE 1.016609519675926, swapped-source MAE 4.386760040509259. All favor correct RGB. New dataset footprint about 45MB; raw source JPEGs and old datasets remain unchanged.

## Execution constraints

ALOHA simulation URDF nominal joint bounds are ±10rad and velocity limits 1000rad/s; passing those checks does not guarantee plausible motion. The 2rad/s trial adds a real per-60ms target-step bound. Original expert nominal per-joint 99th percentile velocities range 0.18–1.94rad/s; boundary timing uncertainty is documented in `expert_nominal_velocity_statistics.json`. A real-joint-state trial changes only feedback state and does not substitute expert actions.

Expert replay with the 2rad/s cap also passes official success at 105 targets. Therefore failure of the capped model rollout is not evidence that the cap itself makes this source task impossible.

## Persistence baseline

On the original full-window subset, holding the current joints is much worse than model prediction: train step-1 L/R joint MAE 0.01221/0.01262rad versus model 0.00191/0.00183; validation persistence 0.01183/0.01301 versus model 0.00371/0.00430. Small one-step errors are not merely a stationary-action artifact. Additional persistence columns are included in the CSV.

## 最终判断

建议重新微调独立 B 对照模型，以修复已证实的监督覆盖缺陷，而非简单增加 A 的步数。正确 RGB 与完整真实观测版本已验证；B 从官方 N1.7 基础权重训练，A 保留。B 同时修正两类数据缺陷，因此它是恢复对照，不是严格单因素 RGB 实验。真实成功仍未实现，B 的成功率也未获验证。


## B 训练与闭环验证完成（2026-10-10）

B 从官方 N1.7 基础权重完成 2000 步，训练运行时间 2717.479 秒，最终记录 loss 0.0206。权重已通过完整性检查和逐文件 SHA256 持久复制校验。正确 RGB 与末段采样覆盖同时修复，因此不是单因素 RGB 对照。末段验证集第 1 步夹爪 MAE 左 0.002255、右 0；TCP 误差左 1.054 mm、右 2.054 mm。

真实模型在训练场景 seed 1 通过官方成功判定：135 个动作，锅高度从 0.779142 m 到 0.869110 m，未调用专家。独立种子 200000–200019，固定执行块 5、去噪步 4、60 ms 动作周期，20 次实际执行，0 成功、20 失败，正式成功率 0%，无初始化/推理中断。训练场景成功不计正式成功率。离线误差改善仍不足以保证新场景闭环成功；剩余原因需结合视频、双臂接触位置和状态分布继续诊断，尚不能归因于训练步数不足。

`videos/01_lift_pot_gr00t.mp4` 为已通过官方判定的真实模型训练场景诊断视频，原始视频保留；来源及计数排除信息见 `evaluation/01_lift_pot_gr00t_provenance.json`。该视频不能证明独立场景成功或稳定泛化。B 离线误差保存在 `results/offline_action_metrics_B.csv` 和 `results/offline_action_metrics_late_phase_B.csv`，A 文件保留。当前尚未达到稳定独立场景成功，不启动多任务扩展，亦未启动任何额外训练。


### B 20 次结果进一步诊断（只读分析，未重复测试或训练）

详见 `results/lift_pot_B_test20_diagnosis.md`。20 次最后观测左右抓取距离均值 24.31/22.94 cm；其中两个掉落中止缺完整终点，18 次可用终点均值 19.02/16.99 cm。输出动作与下发 drive target 完全相同，B 实际 v23 的 150 帧三相机 RGB 抽检全部通过。真实关节跟踪误差早期约 1.5 mm，后期均值 125/172 mm；但 seed 200004 右 TCP 跟踪差仅 2.48 mm，命令 TCP 到抓取点仍至少差 60.78 mm，表明还存在模型抓取目标偏差。50 条原始示范均为 base0，独立测试 base0 12 次和 base1 8 次均失败。只测到一个训练场景成功，不能直接判定参数过拟合或把未见模型解释当成唯一根因。优先逐因素验证真实反馈/接触执行与短动作块，当前没有新增测试或训练；不建议重复已经完成的正确 RGB B 微调。


## B 接触反馈受控实验完成（2026-10-10）

在原始测试划分未训练 base0 种子 48、19、107 上完成 12 次真实模型执行和 3 次同场景源专家回放。固定 B / 60 ms 动作周期 / 去噪 4；drive+h5 1/3（107），实测臂关节+h5 0/3，实测臂关节和夹爪+h5 1/3（19），drive+h1 0/3，专家3/3。反馈显著减少接触后跟踪偏差，但未统一提高成功率，且定位偏差在首次接触已出现。完整反馈的 seed19 结束锅高度 0.820040 m、左右抓取距离 2.844/2.547 cm、朝上 dot 0.9834，两侧末帧指尖接触均存在。基线107也官方成功，但末帧右侧无非零指尖冲量，不声称持续稳定双臂握持。两次成功均为真实模型输出，未调用专家替代；本次未训练或升级，checkpoint保持。原20次正式结果仍0/20，不与本次小规模条件诊断混计。所有15视频全片解码验证通过。详见 results/B_contact_feedback_ablation.md、results/B_contact_feedback_metrics.csv、evaluation/B_contact_feedback/success_provenance.json。
