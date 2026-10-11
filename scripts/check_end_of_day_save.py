"""Read-only weight/video checks and persistent end-of-day manifest."""
import json,struct,hashlib,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'results/end_of_day_2026-10-10';out.mkdir(exist_ok=True)
checkpoints={}
for model,name in [('B','lift_pot_rgb_B'),('C','lift_pot_C_v1')]:
 p=root/f'checkpoints/{name}/checkpoint-2000';idx=json.loads((p/'model.safetensors.index.json').read_text());found={};shards=[]
 for name in sorted(set(idx['weight_map'].values())):
  shard=p/name
  with shard.open('rb') as f:
   n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
  end=max(x['data_offsets'][1] for k,x in header.items() if k!='__metadata__')
  assert shard.stat().st_size==8+n+end
  for k in header:
   if k!='__metadata__':found[k]=name
  shards.append(dict(file=name,bytes=shard.stat().st_size,tensors=len(header)-int('__metadata__' in header)))
 assert found==idx['weight_map']
 required=['config.json','statistics.json','trainer_state.json','experiment_cfg/config.yaml']
 for name in required:assert (p/name).is_file();json.loads((p/name).read_text()) if name.endswith('.json') else None
 assert (p/'processor_config.json').is_file();json.loads((p/'processor_config.json').read_text());json.loads((p/'embodiment_id.json').read_text())
 checkpoints[model]=dict(path=str(p),shards=shards,tensors=len(found),processor_files=['processor_config.json','statistics.json','embodiment_id.json'],complete=True)
records=[];videos=[]
for model in ['B','C']:
 ps=sorted((root/'evaluation/BC_blind_v2').glob(model+'_seed*.json'));assert len(ps)==20
 rs=[json.loads(p.read_text()) for p in ps];assert len({r['seed'] for r in rs})==20
 records.append(dict(model=model,completed=20,official_successes=sum(r['official_success'] for r in rs),hold2s=sum(r['hold_2s_continuous_official'] for r in rs)))
 for r in rs:
  assert (root/f'evaluation/BC_blind_v2/{model}_seed{r["seed"]}_actions.npz').is_file()
  assert Path(r['video']).is_file()
for p in sorted((root/'videos').glob('*.mp4')):
 result=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name,width,height','-of','json',str(p)],capture_output=True,text=True)
 assert result.returncode==0,f'Unreadable video: {p}'
 probe=json.loads(result.stdout);assert float(probe['format']['duration'])>0
 videos.append(dict(file=str(p.relative_to(root)),bytes=p.stat().st_size,probe=probe))
for name in ['results/BC_blind_metrics.csv','results/BC_blind_evaluation.md','experiment_report.md','logs/C-training-v1.log','logs/C-training-gpu-v1.csv','configs/C_blind_evaluation_base0_v2.json','evaluation/C_persistent_copy_verification.json']:
 assert (root/name).is_file() and (root/name).stat().st_size>0
inventory=[]
for directory in ['scripts','configs','results','evaluation','logs']:
 for p in (root/directory).rglob('*'):
  if p.is_file() and '__pycache__' not in str(p):inventory.append(dict(file=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
report=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),checkpoints=checkpoints,paired_evaluation=records,videos=videos,video_check='ffprobe container/stream/duration checks; not full frame decode',mount=subprocess.check_output(['findmnt','-T','/workspace','-n','-o','SOURCE,FSTYPE,OPTIONS'],text=True).strip(),codex_home='/workspace/.codex',inventory=inventory)
(out/'save_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(checkpoints={k:v['tensors'] for k,v in checkpoints.items()},evaluation=records,mp4_count=len(videos)),indent=2))
