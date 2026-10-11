"""Consolidate late-stage evidence into the requested primary CSV, keeping original audit."""
import csv,shutil,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];main=root/'results/offline_action_metrics.csv';backup=root/'results/offline_action_metrics_full_chunks.csv'
if backup.exists():raise FileExistsError('Already consolidated; preserving originals')
shutil.copy2(main,backup);early=list(csv.DictReader(main.open()));late=list(csv.DictReader((root/'results/offline_action_metrics_late_phase.csv').open()));manifest=json.loads((root/'configs/dataset_split_manifest.json').read_text())
for r in early:r['phase_anchor']='full_chunk'
for r in late:r['source_episode']=manifest['splits'][r['split']][int(r['episode'])]['source_episode']
rows=early+late;fields=list(early[0]);
with main.open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print('merged',len(early),len(late),'rows')
