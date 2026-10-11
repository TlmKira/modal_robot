"""Run authorized C training, persist one checked checkpoint and validate inference."""
import hashlib,json,os,shutil,subprocess,time,traceback
from pathlib import Path
root=Path(__file__).resolve().parents[1]
os.chdir(root)
statusfile=root/'results/C_training_status.json'
def status(stage,**fields):
    statusfile.write_text(json.dumps(dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**fields),indent=2))
    print(stage,fields,flush=True)
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
try:
    started=time.monotonic()
    with (root/'logs/C-training-v1.log').open('x') as log:
        train=subprocess.Popen(['bash','scripts/finetune_lift_pot_C_proposed.sh'],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,APPROVED_C_TRAINING='yes'))
        status('training',training_pid=train.pid,max_steps=2000)
        monitor=subprocess.Popen(['python','scripts/monitor_gpu.py','--pid',str(train.pid),'--output','logs/C-training-gpu-v1.csv'])
        code=train.wait()
        monitor.wait(timeout=15)
    assert code==0,f'Training exited {code}; inspect logs/C-training-v1.log'
    source=Path('/tmp/gr00t_C_v1_training/checkpoint-2000')
    state=json.loads((source/'trainer_state.json').read_text())
    assert state['global_step']==2000
    index=json.loads((source/'model.safetensors.index.json').read_text())
    assert all((source/p).is_file() for p in set(index['weight_map'].values()))
    status('persisting_checkpoint',training_seconds=time.monotonic()-started,tensors=len(index['weight_map']))
    target=root/'checkpoints/lift_pot_C_v1/checkpoint-2000'
    staging=target.parent/'.checkpoint-2000.copying'
    assert not target.exists() and not staging.exists()
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(source,staging)
    checks=[]
    for p in source.rglob('*'):
        if p.is_file():
            relative=p.relative_to(source);sha=digest(p)
            assert digest(staging/relative)==sha
            checks.append(dict(file=str(relative),sha256=sha,bytes=p.stat().st_size))
    staging.rename(target)
    (root/'evaluation/C_persistent_copy_verification.json').write_text(json.dumps(checks,indent=2))
    status('validating_real_inference',checkpoint=str(target))
    with (root/'logs/C-policy-check-v1.log').open('x') as log:
        subprocess.run([str(root/'.venvs/gr00t/bin/python'),'scripts/check_finetuned_policy.py','--checkpoint',str(target),'--dataset','datasets/lerobot/lift_pot_C_validation_v1','--output-prefix','C_v1_checkpoint'],stdout=log,stderr=subprocess.STDOUT,check=True)
    status('trained_and_inference_verified_evaluation_pending',checkpoint=str(target),training_seconds=time.monotonic()-started,global_step=2000)
except Exception:
    status('failed_preserving_artifacts',error=traceback.format_exc())
    raise
