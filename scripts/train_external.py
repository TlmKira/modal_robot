"""Prepare an isolated N1.7 run on another GPU host. Dry run is the default.

Does not install dependencies, download models, alter normalization, or resume B/C.
"""
import argparse,json,os,shutil,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--base-model',type=Path,required=True)
p.add_argument('--dataset',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--python',type=Path,default=root/'.venvs/gr00t/bin/python')
p.add_argument('--max-steps',type=int,default=2000)
p.add_argument('--micro-batch',type=int,default=4)
p.add_argument('--accumulation',type=int,default=4)
p.add_argument('--learning-rate',type=float,default=1e-4)
p.add_argument('--persistent-quota-confirmed',action='store_true',help='Confirm personal quota can hold checkpoint rotation; shared df is insufficient')
p.add_argument('--execute',action='store_true',help='Explicitly start GPU training; omit for CPU preflight')
a=p.parse_args()
base=a.base_model.resolve();dataset=a.dataset.resolve();output=a.output.resolve()
if output.exists():p.error('Output must be a new directory; preserving existing runs')
for path in [base/'config.json',base/'model.safetensors.index.json',dataset/'meta/info.json',dataset/'meta/stats.json',dataset/'meta/relative_stats.json']:
 if not path.is_file():p.error(f'Missing required file: {path}')
for value in [a.max_steps,a.micro_batch,a.accumulation]:
 if value<1:p.error('Steps/batch/accumulation must be positive')
if a.learning_rate<=0:p.error('Learning rate must be positive')
parent=output.parent
while not parent.exists():parent=parent.parent
shards=json.loads((base/'model.safetensors.index.json').read_text())['weight_map']
checkpoint_bytes=sum((base/n).stat().st_size for n in set(shards.values()))
required_bytes=2*checkpoint_bytes+5*10**9
space=shutil.disk_usage(parent)
cmd=[str(a.python.resolve()),str(root/'scripts/launch_C_single_export.py'),'--base-model-path',str(base),'--dataset-path',str(dataset),'--embodiment-tag','NEW_EMBODIMENT','--modality-config-path',str(root/'configs/aloha_gr00t_modality.py'),'--output-dir',str(output),'--global-batch-size',str(a.micro_batch),'--gradient-accumulation-steps',str(a.accumulation),'--learning-rate',str(a.learning_rate),'--dataloader-num-workers','2','--shard-size','256','--num-shards-per-epoch','4','--max-steps',str(a.max_steps),'--save-steps','500','--save-total-limit','1','--save-only-model','--no-use-wandb']
print(json.dumps(dict(dry_run=not a.execute,command=cmd,effective_batch=a.micro_batch*a.accumulation,required_rotation_budget_GB=required_bytes/1e9,filesystem_free_GB=space.free/1e9,personal_quota_independently_verified=False,validation='disabled as historical B/C; evaluate held-out actions separately',checkpoint_resume='weights only; optimizer cannot be resumed'),indent=2))
if a.execute:
 if not a.persistent_quota_confirmed:p.error('Confirm personal output quota explicitly before execution')
 if space.free<required_bytes:p.error('Filesystem free space below checkpoint rotation budget')
 subprocess.run(cmd,cwd=root,check=True,env=dict(os.environ,PROJECT_ROOT=str(root)))
