"""Wait for >=50 completed qualified records, preserve outputs, stop surplus collection, build/stat/check only."""
import json,time,os,signal,subprocess,argparse
from pathlib import Path
root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--collector-pids',type=int,nargs=3,required=True);opts=ap.parse_args();pids=opts.collector_pids;folders=['qualified_v1','qualified_worker2','qualified_worker3'];status=root/'results/C_collection_status.json'
while True:
 records=[]
 for label in folders:
  try:records+=json.loads((root/f'datasets/lift_pot_C_{label}/manifest.json').read_text())['records']
  except (FileNotFoundError,json.JSONDecodeError):pass
 qualified=[r for r in records if r.get('accepted') and Path(r.get('file','missing')).is_file()]
 status.write_text(json.dumps({'phase':'collection','completed_records':len(records),'qualified':len(qualified),'target':50,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},indent=2))
 if len(qualified)>=50:break
 live=[]
 for pid in pids:
  try:os.kill(pid,0);live.append(pid)
  except ProcessLookupError:pass
 if not live:raise RuntimeError('All collectors exited before 50 qualified demonstrations')
 time.sleep(10)
# SIGTERM leaves published immutable per-episode files/manifests intact; current unpublished attempts are excluded.
for pid in pids:
 try:
  cmdline=Path(f'/proc/{pid}/cmdline').read_bytes()
  if b'C_collect_qualified.py' in cmdline:os.kill(pid,signal.SIGTERM)
 except (ProcessLookupError,FileNotFoundError):pass
chosen=sorted(qualified,key=lambda r:r['seed'])[:50];seeds={r['seed'] for r in chosen};assert not any(410000<=seed<=410019 for seed in seeds)
(root/'evaluation/C_qualified_union.json').write_text(json.dumps({'accepted':50,'records':[r for r in records if not r.get('accepted') or r['seed'] in seeds],'selection':'first 50 published fully QC-qualified records, ordered by seed; surplus/unpublished not training','collector_stop':'surplus collection terminated after target achieved, no files deleted'},indent=2))
status.write_text(json.dumps({'phase':'dataset_build_and_train_only_stats','qualified':50},indent=2))
with (root/'logs/C-dataset-preparation.log').open('w') as log:subprocess.run(['bash',str(root/'scripts/prepare_C_after_collection.sh')],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
status.write_text(json.dumps({'phase':'prepared_no_training','qualified':50,'training_started':False},indent=2))
