"""Finish seed setup and execute approved fixed B/C evaluation without live watching."""
import json,subprocess,time,traceback,csv
from pathlib import Path
root=Path(__file__).resolve().parents[1]
statuspath=root/'results/BC_blind_status.json'
server=None

def status(stage,**fields):
 statuspath.write_text(json.dumps(dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields),indent=2));print(stage,fields,flush=True)
try:
 status('waiting_for_frozen_base0_seeds')
 deadline=time.monotonic()+3600
 while not (root/'configs/C_blind_evaluation_base0_v2.json').exists():
  if time.monotonic()>deadline:raise RuntimeError('Seed selection timeout; inspect log')
  time.sleep(10)
 cfg=json.loads((root/'configs/C_blind_evaluation_base0_v2.json').read_text());assert len(cfg['seeds'])==20 and len(set(cfg['seeds']))==20
 for model,key in [('B','baseline_B'),('C','C')]:
  status('loading_model',model=model,checkpoint=cfg[key])
  with (root/f'logs/BC-blind-{model}-server.log').open('x') as log:
   server=subprocess.Popen([str(root/'.venvs/gr00t/bin/python'),'-u','scripts/serve_gr00t_local.py','--checkpoint',cfg[key],'--port','6011','--denoising-steps','4'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
   deadline=time.monotonic()+600
   while '"ready": true' not in (root/f'logs/BC-blind-{model}-server.log').read_text():
    if server.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Model server failed to start')
    time.sleep(5)
   status('evaluating',model=model,seeds=cfg['seeds'],controller='drive_h5')
   with (root/f'logs/BC-blind-{model}-rollouts.log').open('x') as runlog:
    subprocess.run([str(root/'.venvs/robotwin/bin/python'),'-u','scripts/BC_blind_runner.py','--model',model,'--port','6011'],cwd=root,stdout=runlog,stderr=subprocess.STDOUT,check=True)
   server.terminate();server.wait(timeout=30);server=None
 records=[]
 for model in ['B','C']:
  for seed in cfg['seeds']:
   r=json.loads((root/f'evaluation/BC_blind_v2/{model}_seed{seed}.json').read_text());rows=r['telemetry'];first={}
   for side in ['left','right']:
    tick=r['first_contact'].get(side,{}).get('physics_tick');row=next((x for x in rows if x['physics_tick']==tick),None)
    first[side+'_first_contact_distance_m']=row[side+'_grasp_distance_m'] if row else None
   records.append(dict(model=model,seed=seed,official_success=r['official_success'],hold_2s_completed=r['hold_2s_completed'],hold_2s_continuous_official=r['hold_2s_continuous_official'],hold_2s_both_contact_seconds=r['hold_2s_both_contact_seconds'],left_tcp_tracking_mean_m=sum(x['left_tcp_tracking_m'] for x in rows)/len(rows),right_tcp_tracking_mean_m=sum(x['right_tcp_tracking_m'] for x in rows)/len(rows),max_pot_height_m=max(x['pot_height_m'] for x in rows),final_pot_up_dot=rows[-1]['pot_up_dot'],both_contact_seconds=r['contact_ticks']['both']*.004,inference_mean_seconds=sum(x['model_seconds'] for x in r['inference'])/len(r['inference']) if r['inference'] else None,error=r.get('error'),video=r['video'],**first))
 with (root/'results/BC_blind_metrics.csv').open('x') as f:
  w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
 summary={m:dict(requested=20,official_successes=sum(r['official_success'] for r in records if r['model']==m),hold_2s_successes=sum(r['hold_2s_continuous_official'] for r in records if r['model']==m),errors=sum(bool(r['error']) for r in records if r['model']==m)) for m in ['B','C']}
 (root/'results/BC_blind_summary.json').write_text(json.dumps(summary,indent=2))
 note='\n\nB/C 固定 drive_h5 全新 base0 对照结果（自动记录）\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n\n官方首次成功与后续 2 秒连续保持分开计数；完整动作和录像位于 evaluation/BC_blind_v2 与 videos/BC_blind_v2_*。\n'
 for name in ['results/BC_blind_evaluation.md','experiment_report.md']:
  with (root/name).open('a') as f:f.write(note)
 status('completed',summary=summary)
except Exception:
 status('failed_preserving_all_artifacts',error=traceback.format_exc());raise
finally:
 if server is not None and server.poll() is None:server.terminate();server.wait(timeout=30)
