import json,csv,subprocess,hashlib,collections
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];union=json.loads((root/'evaluation/C_qualified_union.json').read_text());rows=union['records'];good=[r for r in rows if r.get('accepted')];assert len(good)==50;manifest=json.loads((root/'evaluation/C_dataset_manifest.json').read_text());checks=json.loads((root/'evaluation/C_dataset_loader_check.json').read_text());rgb=json.loads((root/'evaluation/C_dataset_rgb_quality.json').read_text());out=root/'results/C_data_quality';out.mkdir(exist_ok=True)
fields=['seed','source','model_id','accepted','reason','original_official_success','replay_official_success','hold_official_all','hold_both_contact_s','collision_ticks','gripper_closed_final','frames','physics_ticks','file','wall_s']
with (out/'demonstrations.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
covered=np.asarray([r['initial_pose'] for r in good]);x=covered[:,0,3];y=covered[:,1,3];yaw=np.arctan2(covered[:,1,0],covered[:,0,0]);counts=collections.Counter(r.get('reason') for r in rows if not r.get('accepted'));sizes=sum(r.get('bytes',0) for r in good);train=[e for e in manifest if e['split']=='train'];old=[e for e in train if e['source']=='old'];new=[e for e in train if e['source']=='new'];assert len(old)==40 and len(new)==50
protected=json.loads((root/'results/C_preparation/protected_artifacts_manifest.json').read_text());integrity=[]
for r in protected:
 if '/checkpoints/' in r['path'] or '/videos/' in r['path']:
  h=hashlib.sha256()
  with Path(r['path']).open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  integrity.append({'path':r['path'],'unchanged':h.hexdigest()==r['sha256']})
assert all(r['unchanged'] for r in integrity);(out/'B_integrity_recheck.json').write_text(json.dumps(integrity,indent=2))
summary={'new_qualified':50,'source_attempt_records':len(rows),'rejection_counts':dict(counts),'initial_x_range_m':[float(x.min()),float(x.max())],'initial_y_range_m':[float(y.min()),float(y.max())],'yaw_range_rad':[float(yaw.min()),float(yaw.max())],'new_raw_bytes':sizes,'C_training_episodes':90,'C_training_real_anchors':sum(e['real_anchors'] for e in train),'new_real_anchors':sum(e['real_anchors'] for e in new),'old_real_anchors':sum(e['real_anchors'] for e in old),'RGB_first_frame_checks':len(rgb),'RGB_MAE_mean':float(np.mean([r['RGB_MAE'] for r in rgb])),'max_RGB_MAE':max(r['RGB_MAE'] for r in rgb),'statistics_train_only':True,'synthetic_conditioning_anchors':0,'B_checkpoint_videos_unchanged':True};(out/'summary.json').write_text(json.dumps(summary,indent=2));df=subprocess.check_output(['df','-h','/workspace','/'],text=True);du=subprocess.check_output(['du','-sh',str(root/'datasets')],text=True)
text=f'''# C 数据准备验收报告（未开始训练）

新增50条通过完整验收的base0官方专家示范（含此前4条候选），40条历史B训练数据另行保留并纳入独立C版本，共90条训练轨迹。原5条验证和5条历史测试仍分离；未来20种子盲测另锁定 configs/C_blind_evaluation_v1.json。没有训练C、升级框架或覆盖B权重、日志及原数据。

## 质量验收与接口

每条新增示范先通过官方专家任务及关节规划检查，再从相同初始位姿以固定15×4ms接口回放，逐物理步检查官方判定、pot-finger正接触冲量、非抓取碰撞、锅高度和姿态、夹爪命令及实际手指位置。回放完整结束后固定最后关节/夹爪目标、关节速度零，检查2秒保持。全部新增验收通过官方成功、2秒全程官方条件、至少0.8秒双侧有力接触、夹爪命令低于0.1、无超过0.01Ns的检测碰撞。接触冲量阈值1e-8Ns；锅与手指抓取接触允许，锅与机械臂链及动臂对其他实体超过0.01Ns拒绝。低于阈值的瞬时接触不被称为绝对无碰撞。

训练使用绝对drive target，左6/左夹爪/右6/右夹爪；state[t]与action[t+1]严格对应。新增RGB是在真正执行60ms接口的场景中采集，实测qpos及手指米单位单独保存，未混入state。物理tick全局每15步观测，包含真实2.04秒保持段；40步动作窗增加39个未来HOLD，官方sampler未选任何合成观测锚点。原始官方规划动作与接口回放目标、每4ms接触碰撞日志及新HDF5均保留。

新增原始采集是官方Color RGB直接uint8，无红蓝交换；C三相机视频从原RGB重新编码，{len(rgb)} 个首帧抽检全部RGB误差小于交换通道误差，平均MAE {summary['RGB_MAE_mean']:.3f}/255、最大 {summary['max_RGB_MAE']:.3f}/255。该检查范围是每段视频首帧，不冒称所有帧逐像素审计。全部数据通过官方GR00T loader与动作窗检查。

重要历史限制：原40条B源数据没有精确逐帧时间戳，本次保留原state/action序列，按既有约60ms采样语义重新标记时间轴；不能追溯证明所有旧边界间隔均为60ms。新增50条具备精确physics_tick证据。若论文需要全部数据严格物理时序，应另对原40条进行同接口重新采集，而不能伪造缺失历史时间戳。

## 数量、覆盖与拒绝原因

新增成功50条，训练总真实观测锚点 {summary['C_training_real_anchors']}，其中旧 {summary['old_real_anchors']}、新 {summary['new_real_anchors']}。型号全部base0；X范围 {x.min():.4f}–{x.max():.4f}m，Y {y.min():.4f}–{y.max():.4f}m，yaw {yaw.min():.4f}–{yaw.max():.4f}rad。包含双臂接近/对齐、闭合、实际接触、共同抬升及保持。没有未经验证的恢复策略数据，也没有模型失败轨迹监督标签。base1只登记并排除，不进入训练。

完成记录 {len(rows)}，拒绝/跳过原因：{dict(counts)}。达到50个完整发布的验收样本后停止额外采集；最后未发布的进行中尝试不计为成功、不加入训练，已有文件不删除。三个独立进程复用现有GPU，未创建额外付费资源。示范逐条清单 demonstrations.csv，种子与全部日志在 datasets/lift_pot_C_qualified*/。

## 存储、统计及训练预算

50条新增合格原始数据 {sizes/1e9:.3f}GB。当前磁盘：

```text
{df}{du}```

/workspace是网络持久卷；236TB级共享剩余不是个人剩余配额。个人配额/用量尚不可获取，GPU计费单价也未知，不能给出虚构余额/美元价格。原checkpoint及两段模型成功视频重新SHA256核验未变。

C meta/stats.json 与 relative_stats.json 由官方工具仅在新90条训练集重算，验证/测试只复制训练统计并校验字节一致。原B统计未修改。

训练提案：GR00T N1.7官方基础权重起点，沿用B的冻结方案、bf16、有效batch16、lr1e-4、预测horizon40，约2000步，输出全新C目录。保持同起点有利于比较数据扩充效果；从B继续训练会引入累计优化步数混杂，若采用须独立命名C-warm，不能混合比较。现阶段仅准备，不启动100–300步短训或正式训练。短训100–300步建议先用新输出检查显存、loader速度、loss/验证动作误差、checkpoint加载，再精算2000步耗时。

B实测2000步45.3分钟/峰值38.96GB，C粗估45–90分钟加验证与保存，尚非实测。费用=GPU小时×用户实际单价，正式预算需确认单价。新单checkpoint约12GB，根导出另12GB，缓存/保存保守预留30–60GB。

## 统一评测与批准边界

B/C使用固定drive_h5、实际denoising4、60ms执行、同一20个全新未训练未调参种子，禁止按种子切换控制器。官方随机物体型号保留并分层报告base0/base1；base1无训练覆盖应明确注明。记录官方首次成功、首次接触定位、TCP跟踪、锅位姿、双侧接触时间、滑落/碰撞及推理延迟。成功后继续原闭环模型/控制器2秒，1/2秒持续握持分别报告，不能以冻结目标专家保持替代模型保持。原0/20和已有诊断不改写。

请用户确认后才启动C训练。当前所有C产物仅为数据和统计，没有新训练权重。
'''
(out/'C_dataset_readiness.md').write_text(text);print(json.dumps(summary,indent=2))
