"""Finalize measured ablation interpretation, video integrity and success provenance."""
import json,csv,hashlib,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
root=Path(__file__).resolve().parents[1];folder=root/'evaluation/B_contact_feedback';summary=json.loads((root/'evaluation/B_contact_feedback_summary.json').read_text());rows=summary['records'];seeds=[19,48,107];variants=['drive_h5','qpos_arm_h5','qpos_full_h5','drive_h1','expert_replay']
assert len(rows)==15 and {(r['seed'],r['variant']) for r in rows}=={(s,v) for s in seeds for v in variants}
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def video_check(r):
 p=Path(r['video']);probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate,duration','-of','json',str(p)]));subprocess.run(['ffmpeg','-v','error','-i',str(p),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE);return {'seed':r['seed'],'variant':r['variant'],'file':str(p),'sha256':digest(p),'decode_ok':True,'probe':probe}
with ThreadPoolExecutor(max_workers=3) as pool:videos=list(pool.map(video_check,rows))
(folder/'video_checks.json').write_text(json.dumps(videos,indent=2))
checks=json.loads((root/'evaluation/rgb_B_persistent_copy_verification.json').read_text());cp=root/'checkpoints/lift_pot_rgb_B/checkpoint-2000';assert all(digest(cp/x['file'])==x['sha256'] for x in checks if x['file'].endswith('.json'))
manifest=json.loads((root/'configs/dataset_split_manifest.json').read_text());trained={e['source_seed'] for e in manifest['splits']['train']};assert not trained & set(seeds)
initial_diffs={}
for s in seeds:
 x=np.load(folder/f'drive_h5_seed{s}_actions.npz')['predictions'][0];y=np.load(folder/f'drive_h1_seed{s}_actions.npz')['predictions'][0];initial_diffs[s]=float(np.abs(x-y).max());assert initial_diffs[s]==0
success=[]
for r in rows:
 if not r['official_success'] or r['expert_diagnostic_only']:continue
 detail=json.loads((folder/f"{r['variant']}_seed{r['seed']}.json").read_text());final=detail['telemetry'][-1];assert final['official_success'] and final['pot_height_m']>.82 and max(final['left_grasp_distance_m'],final['right_grasp_distance_m'])<.03 and final['pot_up_dot']>.8
 success.append({'seed':r['seed'],'variant':r['variant'],'source_split':'original demonstration test, excluded from B training','source_episode':detail['initial']['source_episode'],'checkpoint':detail['checkpoint'],'checkpoint_config_sha256':detail['checkpoint_config_sha256'],'model':'GR00T N1.7 B checkpoint-2000','policy_source':'live Gr00tPolicy.get_action','expert_actions_executed':False,'official_check':'RoboTwin envs/lift_pot.py check_success','official_success':True,'actions':r['actions'],'physical_final':final,'physical_both_finger_contact_seconds':r['physical_both_finger_contact_s'],'control':{'state_mode':r['state_mode'],'execution_horizon':r['execute_horizon'],'physics_steps_per_action':15,'physics_dt_s':.004,'interpolation_start':'drive target','denoising_steps':4,'ensemble':1},'video':r['video'],'video_sha256':next(v['sha256'] for v in videos if v['seed']==r['seed'] and v['variant']==r['variant']),'action_log':str(folder/f"{r['variant']}_seed{r['seed']}_actions.npz")})
(folder/'success_provenance.json').write_text(json.dumps({'real_model_successes':success,'video_integrity_checked':15,'checkpoint_json_hashes_unchanged':True,'drive_h5_vs_h1_first_prediction_max_abs_difference':initial_diffs},indent=2))
p=root/'results/B_contact_feedback_ablation.md';text=p.read_text();lead=['## 实测结论（15/15 完成）','',
'**现有 B 已在未训练的 base0 场景通过官方 Lift Pot 判定：完整实测反馈在 seed 19 成功；drive-state 基线在 seed 107 成功。没有训练、升级或改变 checkpoint。完整反馈没有提高这 3 个配对场景的总成功数（两者均 1/3），而是交换了成功场景，不能认定为普适改进。**','',
'| 方案 | 诊断成功 | 成功种子 |','|---|---:|---|']
for v in variants:
 rr=[r for r in rows if r['variant']==v];passed=[r['seed'] for r in rr if r['official_success']];lead.append(f"| {v} | {len(passed)}/3 | {', '.join(map(str,passed)) or '无'} |")
lead+=['','1. **真实关节反馈改善跟踪，但不是充分条件。** 失败基线 seed 48 接触后左右 TCP 跟踪均差约 31/59 mm，seed 19 为 29/129 mm；仅实测臂关节使三场景接触后均差降至约 2.6–4.5 mm，但仍 0/3。完整反馈接触后约 2.7–5.0 mm，只有 seed 19 成功。这支持 command-state 造成接触后误差累积，也说明抓取定位与夹爪阶段仍是限制因素。','',
'2. **缩短到 1 步未改善成功。** 同种子、同 drive 输入、同 60 ms 周期、同模型随机种子；各场景首块预测逐位相同（最大差 0），因此没有初始场景/模型加载混杂。1 步组全部失败，且接触后均差仍约 19–92 mm，不能假设更频繁重规划自然更好。该对比仅测试 drive 输入，不外推“完整实测反馈＋1 步”的效果；该组合未运行。推理中位数约 150 ms，1 步回合消耗更多推理墙钟时间。原动作的模拟时间未缩放，但当前推理速度不足以在真实硬件保证 60 ms 同步控制。','',
'3. **首次接触时已经有定位偏差。** 专家三场景首次接触左右距离约 1.16–1.25 / 0.27–0.29 cm，接触后 TCP 跟踪约 2–3 mm，全部成功。失败 drive 基线 seed 48 首次接触为 4.32/6.48 cm，seed 19 为 2.74/6.04 cm，均先出现至少一侧超 3 cm，再发生跟踪发散。1 步组首次接触右臂均为 5.56–6.74 cm。仅实测反馈可消除大部分后期执行偏差，却常停留在错误抓取位置。首次碰到锅并非成功握住指定把手；距离条件也不是任何首次接触必需的阈值，基线 seed 107 初次接触较远，随后仍能靠近并成功。','',
'4. **夹爪尺度与语义已核验。** 输入反馈为物理 base 指关节按控制 scale 反算的命令等价值，两个实际指关节米值另存。机械 q=0 对应等价值 0.181818；此时可已到闭合下限。专家 seed 48 成功时命令 0、实测等价值约 0.239，说明物体阻挡与负值驱动死区使物理状态无法与训练命令逐项等价。seed 19 实测方案的成功不是“归一化已消除语义差异”的证明；它只说明该明确记录的替换在这个场景有效。百分位归一化沿用原 processor，没有重算统计、改变 action 的相对/绝对定义或叠加状态。','',
'5. **覆盖不足是可能的残余因素，尚未证实唯一根因。** 本组没有 base1 模型差异，三条匹配专家均成功，故物体模型未见/场景不可行不能解释本组失败。模型出现不同于示范的碰撞、半闭合和停滞状态，且多组有明显状态百分位范围超界；支持闭环状态覆盖不足的判断，但没有训练对照验证，不能直接宣布必须加数据或加训练步数。只读取测试划分用于本次诊断，样本少且选自专家可行场景，不能混入先前 0/20 或作为新的正式成功率。','',
'## 两次真实模型成功的核验','',
'| 场景 / 方案 | 动作数 | 锅高度 m | 左/右抓取距 cm | 锅朝上 dot | 双指接触累计 s |','|---|---:|---:|---|---:|---:|']
for item in success:
 f=item['physical_final'];lead.append(f"| {item['seed']} / {item['variant']} | {item['actions']} | {f['pot_height_m']:.6f} | {f['left_grasp_distance_m']*100:.3f}/{f['right_grasp_distance_m']*100:.3f} | {f['pot_up_dot']:.4f} | {item['physical_both_finger_contact_seconds']:.3f} |")
lead+=['',
'[优先查看 seed 19 完整实测反馈成功视频](../videos/B_contact_feedback_qpos_full_h5_seed19_raw.mp4)：结束时两侧指尖—锅接触均为真，累计双侧接触 1.348 s；模型真实输出并执行，未用专家替代。seed 107 基线也通过官方判定，累计双侧接触 0.128 s，但末帧右侧没有非零指尖接触冲量，不能据此宣称持续双臂稳固握持。两者均按首次官方成功立即结束，未另行验证成功后长时间保持。','',
'15 个原始视频均已 ffprobe 和全片解码检查；成功来源、配置、物理状态、动作日志及视频 SHA256 保存在 `evaluation/B_contact_feedback/success_provenance.json`。B checkpoint 的 10 个 JSON 文件与训练完成时持久复制校验一致；未写入模型目录。原 20 次正式评测仍为 0/20，未重复启动。','',
'初期在 seed 200000 尝试新专家规划失败，未启动该场景模型回合；预检日志/失败记录保留，排除于本表。随后按用户“已有数据对应”条件改用原始测试划分 seed 48/19/107，直接读取相同源专家轨迹。','',
'**下一步判断：无需新训练即可获得真实未见 base0 成功，但尚无在这 3 场景稳定胜出的统一状态/执行方案。保留 drive＋5 步作为兼容基线及完整反馈＋5 步作为受控候选；不按种子切换方案冒充统一策略。当前不启动新训练、升级或完整 20 次重测。**','',
'以下为方法、逐回合指标、曲线与视频。','']
# Keep generated measured methods/table and prepend interpretation.
text=text.replace('本文件由实测数据生成；所有组尚未齐全时只报告已完成项，不推断缺失结果。','15/15 回合已完成；其中 12 次模型执行、3 次独立标记的专家回放。成功回合提前结束，失败为 200 动作上限，各组后接触时长不同，平均跟踪误差属于描述性统计。')
p.write_text(text.split('\n',1)[0]+'\n\n'+'\n'.join(lead)+text.split('\n',1)[1])
print(json.dumps({'groups':summary['groups'],'successful_real_model_runs':len(success),'verified_videos':len(videos)},indent=2))
