# C代码与控制审计（2026-10-11）

## 范围与保存状态

阅读README、README_RESUME、实验报告、旧根因/15次反馈诊断、B/C实际checkpoint小型JSON/YAML配置和40条正式结果。只使用Python标准库和NumPy，不导入torch、SAPIEN、GR00T模型，不加载权重。B/C各3分片与config、processor、statistics、embodiment文件存在；40条结果引用的全部MP4存在。模型9/20与保持6/20不变。

复现：`.venvs/data_conversion/bin/python scripts/audit_C_cpu_history.py`。机器可读证据：`results/C_cpu_audit_2026-10-11/evidence.json`。该脚本仅读取小型元数据、JSON及NPZ动作，生成独立审计文件；不改控制、数据、checkpoint或历史指标。

六个关键源码（runner、telemetry、adapter、server、modality、executor）SHA256与昨天收工manifest一致，**没有发现后续代码回归**。README仍是早期状态，写着尚无成功，应以恢复文档和正式汇总为准；本轮不改历史README。

## 九项边界检查

|项目|代码位置|证据与结论|
|---|---|---|
|三路RGB|scripts/gr00t_robotwin_interface.py:9–19；configs/aloha_gr00t_modality.py:10；RoboTwin/envs/camera/camera.py:273、323|head→head_camera，left_wrist→left_camera，right_wrist→right_camera；get_obs先update_render/update_picture。相机依次拍照但中间不scene.step；同一物理时刻。RGBA截RGB，uint8[0,255]，adapter无BGR交换。CPU彩色哨兵通过，不重复昂贵视频验证。|
|图像预处理|adapter及checkpoint processor_config|推理将HWCuint8组为[B,T,H,W,C]，由同一已保存processor处理。历史RGB正确验证保留；未验证后处理实时设备时间戳，因为日志未存输入图像时间戳。|
|状态定义|RoboTwin robot.py:494–524；BC_blind_runner.py:44–46；build_C_dataset.py|train/online均drive14维。视觉实际、drive命令可分离，是设计局限；直接替qpos破坏语义。|
|14维及归一化|adapter:8、23–42；checkpoint statistics；GR00T policy.py:424、state_action_processor.py:478–516|左6/左夹爪/右6/右夹爪；关节rad、夹爪[0,1]；模型内部relative与百分位归一化，decode返回反归一化absolute。adapter不再叠加state、不二次归一化，CPU哨兵通过。|
|relative→absolute|官方decode_action与runner执行|参考state为本次观测最后状态，各未来步相对同一参考，不是逐步累加。更换观测state也会更换decode参考，风险不止视觉融合变化。|
|60ms执行|BC_blind_runner.py:20–30；RoboTwin_base_task.py:224|每action15次physics step，默认dt=.004；arm速度=(target-drive)/.06，线性插值，双侧在同tick步进。纯CPU mock证实15tick、最终目标与速度正确。需未来加入dt断言，避免环境配置变化。推理期间物理暂停；h5是每300ms模拟时间规划，墙钟推理约150ms，不是实时硬件60ms能力。|
|夹爪|robot.py:508、618–664；BC_blind_telemetry.py:7–25|普通real_state及get_normal_real_gripper_val并非完整真实夹爪读数，后者读drive target；日志单独从实体qpos读取mimic/base指。command、drive、real必须分开命名。set_gripper先缓存请求值，可能用eps=.1限目标，理论可命令≠drive，但40日志最大差3.25e-8，未证实本轮限幅损害。|
|reset/seed|BC runner:35–42、finally；Base_Task init:59–130；server reset；GR00T reset:482|新task/scene，初始化home/open、物体与计数，模型torch seed同seed；Gr00tPolicy.reset本身为空，但当前单帧policy未见跨回合缓存。20对初始锅/drive/real差0，跨C episode初始drive相同。Python random seed被官方注释，所走路径可比但未来需记录随机源，不能盲目改seed顺序。|
|输出→执行|adapter；runner:54–62；NPZ predictions/executed|40条中执行序列与每次预测前5步截取后的前缀逐项完全一致（max_abs=0）。这是请求目标一致，不能据此宣称物理qpos完全跟踪；实测另见telemetry。|

夹爪机械零位不等于归一化命令0：ALOHA scale负端与物理限位有死区。不能把实际等价值0.18理解为仍开18%，不能随意把real输出夹爪直接喂训练命令字段。

## 6+5+3具体失败

分类基于已保存60ms遥测及接触事件，官方判定实际逐4ms检测；“未同时对齐”只表示采样中未见，不能排除瞬时对齐。

- **6次未同时对齐**：410001、410005、410019、410021、410026、410028。410001最佳同刻实际最大距离4.57cm，drive目标对应4.40cm、L/R跟踪约2.7/4.5mm，定位/目标生成偏差比执行误差更直接；不能进一步证明必是视觉误差。410019最佳时实际4.63cm、左跟踪5.48cm；410021实际4.92cm、左跟踪4.44cm，接触执行偏差明显。不能把六条统一归于模型定位。
- **5次对齐未有效抬升**：410010、410013、410015、410016、410025。最佳双侧距离约1.39–2.10cm；相应目标—真实跟踪多数毫米、410016左约9.7mm。410025最佳时L/R跟踪约2.8/2.8mm，仍未在对齐时抬升过0.82m，反证“只修关节跟踪就一定成功”。接触累计不等于握把，低位夹持、夹爪、双臂施力/高度时序均可能，缺少完整姿态/力/碰撞数据不能定唯一根因。
- **3次官方成功后保持失败**：410006、410009、410012。410006始终双侧受力接触约2秒，末帧仍满足官方条件，但保持窗口中曾短暂失效；不是必然滑落。410009末端左右距约24.8cm、height.81875m、up_dot=-.45073，左跟踪6.61cm，明确发生姿态失稳/翻转与接触丢失，但先后因果未确定。410012末端左距4.382cm、左跟踪4.06mm、height.8764m/up_dot.99936，机器人基本追踪命令却保持几何失败，更支持目标/相对位姿问题。

因此定位、双臂相对运动、接触后跟踪和抓持物理均有迹象，但没有从日志证实相机回归、14维映射错、反归一化错或reset继承末态。

## 已证实的Bug与已证实的局限

1. **接触时长off-by-one**：BC runner的hold窗口[first,first+500]包含501个tick，再乘dt，最大2.004秒。CPU改统计为(first,first+500]得到2.000秒，6/20连续官方保持判定不变。仅修指标，不改变模型成功率；原指标不覆盖，校正结果保存在独立evidence。
2. **反馈命名与读取局限**：官方get_normal_real_gripper_val取drive，real_jointState夹爪取command。源码确定，但不证明本轮驱动有Bug；项目telemetry真实指反馈正确单独保存。未来调用方误当真实反馈会出错，需语义化接口/断言。
3. **评测口径差异**：没有expert_check，且base0限定、自定义60ms执行、200步上限、逐4ms官方判定，均应与官方默认分开标注；不是模型Bug，不能事后筛掉失败。
4. **可观测性不足**：保持bool每4ms保存但完整状态通常60ms；没有每失效tick的完整pose、qvel、命令/drive/实际指位置，也没有全机器人非抓取碰撞分类。故之前nearest样本可能仍success，不能精确诊断瞬时违规。
5. **异常清理潜在缺口**：setup_demo/Recorder构造在rollout try/finally之外，初始化抛错时未保证close；40条无错误，无已证实成功率影响。宜新runner保护初始化，不能称本轮失败由此导致。

## 证据分级

- 已证实代码统计Bug：保持接触时长起始tick多计。
- 已证实物理/反馈差异：drive目标与qpos接触后偏离，部分C失败左跟踪厘米级；并非所有失败均有此差异。
- 有证据但非唯一归因：目标定位不准、双臂协调/相对几何失败、持物翻转。
- 仅假设：纯视觉定位错误、摩擦系数不合适、夹爪力不足、示范恢复覆盖不足、过拟合、最优超参数未选到。需要独立对照。
- 当前未见证据：RGB再次反转、动作组顺序错、double-relative-add、reset串回合。

昨天主要训练参数一致且符合官方默认/示例。C验证集完整分阶段离线误差尚缺，不能把旧A/B低误差外推到C。新增数据既改变量，也改变稳定性及时间质量，需保留这一混杂限制。
