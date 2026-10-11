"""Summarize only measured paired contact ablations, including incomplete batches."""
import json,csv,argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--run-label',default='');args=ap.parse_args();prefix='B_contact_feedback'+('_'+args.run_label if args.run_label else '')
root=Path(__file__).resolve().parents[1];folder=root/'evaluation'/prefix;results=[];series=[]
for path in sorted(folder.glob('*_seed*.json')):
 if path.name.startswith('expert_generation'):continue
 r=json.loads(path.read_text());ts=r['telemetry'];first=min((v['sim_time_s'] for k,v in r['first_contact'].items() if k in ['left_arm','right_arm']),default=None);pre=[x for x in ts if first is None or x['sim_time_s']<first];post=[x for x in ts if first is not None and x['sim_time_s']>=first];end=ts[-1]
 row={'seed':r['seed'],'variant':r['variant'],'expert_diagnostic_only':r['expert_diagnostic_only'],'state_mode':r['state_mode'],'execute_horizon':r['execute_horizon'],'official_success':r['success'],'actions':r['actions'],'sim_duration_s':r['physics_ticks']*.004,'first_robot_pot_contact_s':first,'first_left_finger_contact_s':r['first_contact'].get('left',{}).get('sim_time_s'),'first_right_finger_contact_s':r['first_contact'].get('right',{}).get('sim_time_s'),'best_simultaneous_max_distance_m':min(max(t['left_grasp_distance_m'],t['right_grasp_distance_m']) for t in ts),'final_pot_height_m':end['pot_height_m'],'final_pot_up_dot':end['pot_up_dot'],'physical_both_finger_contact_s':r['physical_finger_contact_ticks']['both']*.004,'inference_calls':len(r['inference']),'inference_model_mean_s':float(np.mean([v['model_seconds'] for v in r['inference']])) if r['inference'] else None,'inference_model_median_s':float(np.median([v['model_seconds'] for v in r['inference']])) if r['inference'] else None,'inference_warm_mean_s':float(np.mean([v['model_seconds'] for v in r['inference'][1:]])) if len(r['inference'])>1 else None,'inference_model_p95_s':float(np.percentile([v['model_seconds'] for v in r['inference']],95)) if r['inference'] else None,'inference_total_model_s':sum(v['model_seconds'] for v in r['inference']),'state_outside_q01_q99_fraction':r['clipped_state_components']/r['policy_state_components'] if r['policy_state_components'] else None,'error':r.get('error',''),'video':r['video']}
 for side in ['left','right']:
  row[f'pre_contact_last_{side}_grasp_distance_m']=pre[-1][f'{side}_grasp_distance_m'] if pre else None
  row[f'at_first_contact_{side}_grasp_distance_m']=post[0][f'{side}_grasp_distance_m'] if post else None
  row[f'post_contact_{side}_tcp_tracking_m_max']=max((x[f'{side}_tcp_tracking_m'] for x in post),default=None)
  row[f'post_contact_{side}_tcp_tracking_m_p95']=float(np.percentile([x[f'{side}_tcp_tracking_m'] for x in post],95)) if post else None
  for metric in ['grasp_distance_m','drive_tcp_grasp_distance_m','tcp_tracking_m','joint_mae_rad','joint_max_error_rad','gripper_command','gripper_drive_normalized','gripper_actual_normalized']:
   key=f'{side}_{metric}';row[f'final_{key}']=end[key]
  row[f'min_{side}_grasp_distance_m']=min(x[f'{side}_grasp_distance_m'] for x in ts)
  row[f'final_{side}_gripper_base_actual_m']=end[f'{side}_finger_actual_normalized'][0]*.055-.01
  row[f'final_{side}_gripper_mimic_actual_m']=end[f'{side}_finger_actual_normalized'][1]*.055-.01
  row[f'final_{side}_gripper_at_lower_stop']=all(x*.055-.01<=1e-4 for x in end[f'{side}_finger_actual_normalized'])
  row[f'{side}_physical_finger_contact_s']=r['physical_finger_contact_ticks'][side]*.004
  for phase,subset in [('pre_contact',pre),('post_contact',post)]:
   for metric in ['grasp_distance_m','tcp_tracking_m','joint_mae_rad']:
    row[f'{phase}_{side}_{metric}_mean']=float(np.mean([x[f'{side}_{metric}'] for x in subset])) if subset else None
   row[f'{phase}_{side}_min_grasp_distance_m']=min((x[f'{side}_grasp_distance_m'] for x in subset),default=None)
 for t in ts:
  for side in ['left','right']:
   finger_m=[x*.055-.01 for x in t[f'{side}_finger_actual_normalized']]
   t[f'{side}_gripper_base_actual_m']=finger_m[0];t[f'{side}_gripper_mimic_actual_m']=finger_m[1];t[f'{side}_gripper_at_lower_stop']=all(x<=1e-4 for x in finger_m)
  series.append({'seed':r['seed'],'variant':r['variant'],'phase':'post_contact' if first is not None and t['sim_time_s']>=first else 'pre_contact',**{k:v for k,v in t.items() if not isinstance(v,(list,dict))}})
 if not r['expert_diagnostic_only']:
  state_stats=json.loads((root/'checkpoints/lift_pot_rgb_B/checkpoint-2000/statistics.json').read_text())['new_embodiment']['state']
  groups=['left_arm','left_gripper','right_arm','right_gripper'];lo=np.concatenate([state_stats[k]['q01'] for k in groups]);hi=np.concatenate([state_stats[k]['q99'] for k in groups]);state=np.load(folder/f"{r['variant']}_seed{r['seed']}_actions.npz")['policy_states']
  excess=np.maximum(np.maximum(lo-state,state-hi),0)/np.maximum(hi-lo,1e-8)
  row['state_excess_over_1percent_range_fraction']=float(np.mean(excess>.01));row['state_max_excess_fraction_of_range']=float(excess.max())
 else:row['state_excess_over_1percent_range_fraction']=None;row['state_max_excess_fraction_of_range']=None
 results.append(row)
if not results:raise RuntimeError('No completed instrumented rollouts yet')
for filename,data in [(f'{prefix}_metrics.csv',results),(f'{prefix}_timeseries.csv',series)]:
 with (root/'results'/filename).open('w') as stream:w=csv.DictWriter(stream,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
summary={v:{'episodes':len([r for r in results if r['variant']==v]),'successes':sum(r['official_success'] for r in results if r['variant']==v)} for v in sorted({r['variant'] for r in results})}
(root/f'evaluation/{prefix}_summary.json').write_text(json.dumps({'groups':summary,'records':results},indent=2))
fig,axes=plt.subplots(3,3,figsize=(15,11))
colors={'drive_h5':'C0','qpos_arm_h5':'C1','qpos_full_h5':'C2','drive_h1':'C3','expert_replay':'C4'}
seeds=sorted({r['seed'] for r in results})
for col,seed in enumerate(seeds):
 for r in results:
  if r['seed']!=seed:continue
  ticks=[t for t in series if t['seed']==seed and t['variant']==r['variant']];x=[t['sim_time_s'] for t in ticks];color=colors[r['variant']]
  axes[0,col].plot(x,[max(t['left_grasp_distance_m'],t['right_grasp_distance_m'])*100 for t in ticks],color=color,label=r['variant'])
  axes[1,col].plot(x,[max(t['left_tcp_tracking_m'],t['right_tcp_tracking_m'])*100 for t in ticks],color=color)
  axes[2,col].plot(x,[t['pot_height_m'] for t in ticks],color=color)
 axes[0,col].set_title(f'Seed {seed} / base0');axes[0,col].axhline(3,color='red',ls='--');axes[0,col].legend(fontsize=7);axes[2,col].axhline(.82,color='red',ls='--')
for i,label in enumerate(['max(L,R) contact distance (cm)','max(L,R) drive-real TCP (cm)','Pot height (m)']):
 for ax in axes[i]:ax.set(ylabel=label,xlabel='Simulated seconds');ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(root/f'results/{prefix}_curves.png',dpi=150)
fig,axes=plt.subplots(4,3,figsize=(15,13))
for col,seed in enumerate(seeds):
 for r in results:
  if r['seed']!=seed:continue
  ticks=[t for t in series if t['seed']==seed and t['variant']==r['variant']];x=[t['sim_time_s'] for t in ticks]
  for i,key in enumerate(['left_grasp_distance_m','right_grasp_distance_m','left_tcp_tracking_m','right_tcp_tracking_m']):axes[i,col].plot(x,[t[key]*100 for t in ticks],color=colors[r['variant']],label=r['variant'])
 axes[0,col].set_title(f'Seed {seed}');axes[0,col].legend(fontsize=6)
for i,label in enumerate(['Left contact distance (cm)','Right contact distance (cm)','Left TCP drive-real (cm)','Right TCP drive-real (cm)']):
 for ax in axes[i]:
  ax.set(ylabel=label,xlabel='Simulated seconds');ax.grid(alpha=.2)
  if i<2:ax.axhline(3,color='red',ls='--')
fig.tight_layout();fig.savefig(root/f'results/{prefix}_LR_curves.png',dpi=150)
lines=['# B 接触反馈与执行块受控诊断','',f'已记录 {len(results)} 个小规模诊断回合；不重复原 20 次正式评测。checkpoint 固定为 `checkpoints/lift_pot_rgb_B/checkpoint-2000`，GR00T N1.7，无训练/升级。', '',
'种子为 48、19、107：均为原始数据测试划分的未训练 base0 场景。每个场景读取对应原始官方专家轨迹，通过相同固定控制接口回放，专家结果不计模型成功。各组启动时物体 pose 逐项验证一致；模型随机种子同场景一致。','',
'训练状态是 drive target，不是实际 qpos。实测臂 qpos 保持 rad 与相同关节顺序，仍使用 B 原有 q01/q99 归一化、相对动作解码；不改统计。`qpos_arm_h5` 只更换 12 个臂关节，夹爪输入保留命令；`qpos_full_h5` 再用物理夹爪关节反算实测开合量。夹爪 base/mimic 各关节按 (q-offset)/multiplier、URDF scale [-0.01,0.045] 反算，保留 base/mimic 两指原始读数，以 base joint 反算值作为单维模型反馈，输入裁剪到 [0,1]；URDF 指关节下限 0 m、上限 0.04765 m，而命令 scale 下限为 -0.01 m；因此物理闭合至 q=0 时，命令尺度等价值为 0.181818，绝不能解释成仍打开 18%。另记录 base/mimic 实测米值和下限状态。握住物体时不能仅凭未完全到机械下限判失败。物理 qpos 与训练 command 状态有分布差别，夹爪物理值不能表达示范中以命令为状态的同一控制阶段，本实验评估该替换效果，不宣称语义等价。超出训练 q01/q99 的比例以及超过范围宽度 1% 的显著超界比例写入 CSV，微小边界噪声不能单凭计数判作严重分布外。state 夹爪 q99=1.0，action 夹爪 q99=0.96231，两者不能混淆。', '',
'动作时间始终是 15×4 ms=60 ms。执行块 5/1 对比只使用 drive 输入，重规划的仿真间隔从 300 ms 改为 60 ms。推理期间不推进物理时钟；因此仿真时序保持一致，但更高推理负担不能声称适用于真实硬件实时 60 ms 控制。执行器插值起点始终为 drive target，未同时更改控制起点。','',
'接触定义：锅 link 与左右臂/指尖 link 的接触 impulse 总量 >1e-8 Ns，排除锅与桌面。逐物理步检测首次接触及接触持续时间；每动作及首次事件保存遥测。FK(real qpos) 与原生 TCP 必须误差 <0.1 mm，否则停止记录为错误；不以接触点距离替代物理接触。','',
'| 种子 | 方案 | 官方成功 | 最佳同时 max(L,R) cm | 终点左/右距离 cm | 首次机械臂接触 s | 推理均时 ms |','|---:|---|---|---:|---|---:|---:|']
for r in results:
 fmt=lambda x:'—' if x is None else f'{x:.3f}'
 lines.append(f"| {r['seed']} | {r['variant']} | {r['official_success']} | {r['best_simultaneous_max_distance_m']*100:.2f} | {r['final_left_grasp_distance_m']*100:.2f}/{r['final_right_grasp_distance_m']*100:.2f} | {fmt(r['first_robot_pot_contact_s'])} | {fmt(r['inference_model_mean_s']*1000 if r['inference_model_mean_s'] is not None else None)} |")
lines+=['','分组计数：`'+json.dumps(summary,ensure_ascii=False)+'`。3 个配对诊断种子不是新的 20 次正式成功率，用于配对诊断，不能视为新的完整 20 种子盲测。','',
'逐回合指标见 [metrics CSV](B_contact_feedback_metrics.csv)，时间序列见 [timeseries CSV](B_contact_feedback_timeseries.csv)。详细物理夹爪、接触、预测/实际命令、checkpoint 与配置日志保存在 `evaluation/B_contact_feedback/`，完整运行日志 `logs/B-contact-feedback-data-seeds.log`。','',
'![接触距离、跟踪偏差及锅高度](B_contact_feedback_curves.png)','', '![左右抓取距离和目标—实际 TCP 跟踪](B_contact_feedback_LR_curves.png)','',
'## 原始真实诊断视频','']
for r in results:lines.append(f"- [{r['variant']} / seed {r['seed']} / success={r['official_success']}](../videos/{Path(r['video']).name})")
lines+=['','本文件由实测数据生成；所有组尚未齐全时只报告已完成项，不推断缺失结果。复现：先用现有 GR00T 环境启动 `serve_gr00t_local.py --checkpoint checkpoints/lift_pot_rgb_B/checkpoint-2000 --port 6009`，再用现有 RoboTwin 环境运行 `B_contact_feedback_runner.py --port 6009`。runner 拒绝覆盖已有结果/视频，复现前须使用独立输出目录版本；不可直接重跑覆盖。汇总：`.venvs/gr00t/bin/python scripts/summarize_B_contact_feedback.py`。']
report='\n'.join(lines)+'\n'
for ending in ['metrics.csv','timeseries.csv','curves.png','LR_curves.png']:report=report.replace('B_contact_feedback_'+ending,prefix+'_'+ending)
report=report.replace('复现前须使用独立输出目录版本','复现时 runner 和汇总均传入相同的新 --run-label（如 replay01）以保存独立版本；')
(root/f'results/{prefix}_ablation.md').write_text(report)
print(json.dumps(summary),flush=True)
