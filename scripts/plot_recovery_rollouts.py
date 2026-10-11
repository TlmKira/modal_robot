"""Plot model feedback diagnostics; official JSON is the only success source."""
import json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];fig,axes=plt.subplots(1,3,figsize=(13,4));count=0
for p in sorted((root/'evaluation').glob('gr00t_lift_pot_recovery*.json')):
 for r in json.loads(p.read_text()):
  if not r.get('completed_rollout',True) or not r.get('valid_factor_comparison',True):continue
  cs=r.get('inference_calls',[])
  if not cs:continue
  count+=1;x=[c['action_count']*0.06 for c in cs];label=f"{r['tag']} seed{r['seed']} {'PASS' if r['success'] else 'FAIL'}"
  for side in ['left','right']:axes[0].plot(x,[c[f'{side}_grasp_distance'] for c in cs],label=label+' '+side)
  axes[1].plot(x,[c['pot_height'] for c in cs],label=label);axes[2].plot(x,[c['max_joint_tracking_error'] for c in cs],label=label)
for ax,title in zip(axes,['TCP to moving grasp point (m)','Pot height (m)','Max drive-to-measured tracking error (rad)']):ax.set(xlabel='Simulated time (s)',ylabel=title);ax.grid();ax.legend(fontsize=5)
axes[0].axhline(.03,color='k',linestyle='--');axes[1].axhline(.82,color='k',linestyle='--');fig.tight_layout();fig.savefig(root/'results/recovery_closed_loop_curves.png',dpi=160);print('plotted',count)
