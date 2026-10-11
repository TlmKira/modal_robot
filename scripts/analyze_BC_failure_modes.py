"""Analyze saved paired rollouts; no simulation, training or test reruns."""
import json,csv
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
rows=[];details={}
for model in ['B','C']:
 for p in sorted((root/'evaluation/BC_blind_v2').glob(model+'_seed*.json')):
  r=json.loads(p.read_text());ts=r['telemetry'];best=min(ts,key=lambda t:max(t['left_grasp_distance_m'],t['right_grasp_distance_m']))
  aligned=[t for t in ts if t['left_grasp_distance_m']<.03 and t['right_grasp_distance_m']<.03]
  first=min([x['physics_tick'] for x in r['first_contact'].values()],default=None)
  pre=[t for t in ts if first is None or t['physics_tick']<first];post=[t for t in ts if first is not None and t['physics_tick']>=first]
  ht=r['hold_telemetry'];bad=next((h for h in ht if not h['official']),None)
  badrow=min(ts,key=lambda t:abs(t['physics_tick']-bad['tick'])) if bad else None
  row=dict(model=model,seed=r['seed'],success=r['official_success'],hold2s=r['hold_2s_continuous_official'],best_simultaneous_max_m=max(best['left_grasp_distance_m'],best['right_grasp_distance_m']),best_left_m=best['left_grasp_distance_m'],best_right_m=best['right_grasp_distance_m'],aligned_samples=len(aligned),aligned_max_height=max((t['pot_height_m'] for t in aligned),default=None),best_up_dot=best['pot_up_dot'],both_contact_s=r['contact_ticks']['both']*.004)
  for side in ['left','right']:
   for stage,group in [('pre',pre),('post',post)]:row[f'{side}_{stage}_tracking_mean_m']=float(np.mean([t[side+'_tcp_tracking_m'] for t in group])) if group else None
  if not r['official_success']:
   if not aligned:row['failure_category']='never_jointly_within_3cm'
   elif not any(t['pot_height_m']>.82 for t in aligned):row['failure_category']='aligned_but_no_lift_above_082'
   else:row['failure_category']='alignment_lift_and_upright_not_simultaneous'
  elif not r['hold_2s_continuous_official']:row['failure_category']='success_then_hold_failure'
  else:row['failure_category']='stable_success'
  rows.append(row)
  details[f'{model}_{r["seed"]}']=dict(first_contact_tick=first,best=best,first_hold_failure_tick=bad,nearest_hold_failure_state=badrow)
with (root/'results/BC_failure_modes.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
summary={}
for model in ['B','C']:
 group=[r for r in rows if r['model']==model];categories={k:sum(r['failure_category']==k for r in group) for k in sorted({r['failure_category'] for r in group})}
 summary[model]=dict(categories=categories,best_simultaneous_max_cm_mean=float(np.mean([r['best_simultaneous_max_m'] for r in group]))*100,median=float(np.median([r['best_simultaneous_max_m'] for r in group]))*100)
(root/'evaluation/BC_failure_analysis.json').write_text(json.dumps(dict(summary=summary,rows=rows,details=details),indent=2))
print(json.dumps(summary,indent=2))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(12,8))
for ax,seed in zip(axes.flat,[410000,410006,410009,410019]):
 for model in ['B','C']:
  r=json.loads((root/f'evaluation/BC_blind_v2/{model}_seed{seed}.json').read_text());ts=r['telemetry']
  ax.plot([t['sim_time_s'] for t in ts],[100*max(t['left_grasp_distance_m'],t['right_grasp_distance_m']) for t in ts],label=model)
 ax.axhline(3,color='k',linestyle='--');ax.set(title=f'Seed {seed}',xlabel='Simulation time (s)',ylabel='Max of L/R grasp distance (cm)',ylim=(0,25));ax.legend();ax.grid()
fig.tight_layout();fig.savefig(root/'results/BC_failure_distance_curves.png',dpi=160)
