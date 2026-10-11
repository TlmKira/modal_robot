import json
from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=root/'results/C_data_quality';out.mkdir(exist_ok=True);good=[r for r in json.loads((root/'evaluation/C_qualified_union.json').read_text())['records'] if r.get('accepted')]
fig,axs=plt.subplots(1,3,figsize=(13,4));positions=np.asarray([r['initial_pose'] for r in good]);axs[0].scatter(positions[:,0,3],positions[:,1,3]);axs[0].set(xlabel='Initial pot x (m)',ylabel='Initial pot y (m)',title='50 qualified base0 scenes');dist=np.asarray([[r['first_contact'][s]['grasp_distance'][i] for i,s in enumerate(['left','right'])] for r in good]);axs[1].plot(dist[:,0]*100,label='left');axs[1].plot(dist[:,1]*100,label='right');axs[1].set(xlabel='Demonstration',ylabel='Grasp distance at first contact (cm)',title='First positive finger contact');axs[1].legend()
for r in good:
 data=json.loads((Path(r['file']).parent/f"telemetry_seed{r['seed']}.json").read_text());last=data[-1]['tick'];hold=[x for x in data if x['tick']>last-500];axs[2].plot([(x['tick']-hold[0]['tick'])*.004 for x in hold],[x['height'] for x in hold],alpha=.35,lw=.7)
axs[2].axhline(.82,c='red',ls='--');axs[2].set(xlabel='Hold time (s)',ylabel='Pot height (m)',title='Independent fixed-target expert hold');fig.tight_layout();fig.savefig(out/'C_quality_curves.png',dpi=180)
