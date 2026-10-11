"""Finish the approved B run and genuine model evaluation without live polling by Codex."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--training-pid', required=True, type=int)
args = parser.parse_args()
os.chdir(root)
status_path = root / 'results/rgb_B_background_status.json'
server = None


def status(stage, **details):
    status_path.write_text(json.dumps({'stage': stage, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **details}, indent=2))
    print(stage, details, flush=True)


def run(arguments, logfile):
    with (root / 'logs' / logfile).open('x') as stream:
        subprocess.run(arguments, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=True)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


try:
    status('waiting_for_approved_training', training_pid=args.training_pid)
    while Path(f'/proc/{args.training_pid}').exists():
        time.sleep(30)
    source = Path('/tmp/gr00t_rgb_B_training/checkpoint-2000')
    status('checking_saved_checkpoint', checkpoint=str(source))
    run(['python', 'scripts/summarize_rgb_B_training.py'], 'rgb-B-checkpoint-audit.log')
    gr00t = str(root / '.venvs/gr00t/bin/python')
    dataset_version = 'rgb_v23_terminal_pad'
    run([gr00t, 'scripts/check_finetuned_policy.py', '--checkpoint', str(source),
         '--dataset', f'datasets/lerobot/lift_pot_validation_{dataset_version}',
         '--output-prefix', 'rgb_B_checkpoint'], 'rgb-B-policy-check.log')
    status('persisting_verified_checkpoint')
    target = root / 'checkpoints/lift_pot_rgb_B/checkpoint-2000'
    staging = target.parent / '.checkpoint-2000.copying'
    assert not target.exists() and not staging.exists(), 'Preserving existing destination'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, staging)
    checks = []
    for path in source.rglob('*'):
        if path.is_file():
            relative = path.relative_to(source)
            original_hash = digest(path)
            assert digest(staging / relative) == original_hash
            checks.append({'file': str(relative), 'sha256': original_hash})
    staging.rename(target)
    (root / 'evaluation/rgb_B_persistent_copy_verification.json').write_text(json.dumps(checks, indent=2))
    status('offline_prediction_metrics', checkpoint=str(target))
    run([gr00t, 'scripts/offline_recovery_metrics.py', '--checkpoint', str(target),
         '--dataset-version', dataset_version, '--output-suffix', '_B'], 'rgb-B-offline-metrics.log')
    run([gr00t, 'scripts/audit_late_phase_actions.py', '--checkpoint', str(target),
         '--dataset-version', dataset_version, '--output-suffix', '_B'], 'rgb-B-late-phase-metrics.log')
    status('loading_real_model_server')
    server_log = (root / 'logs/rgb-B-server.log').open('x')
    server = subprocess.Popen([gr00t, '-u', 'scripts/serve_gr00t_local.py', '--checkpoint', str(target),
                               '--port', '6007', '--denoising-steps', '4'], stdout=server_log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 600
    while '"ready": true' not in (root / 'logs/rgb-B-server.log').read_text():
        if server.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError('B server failed to become ready; inspect logs/rgb-B-server.log')
        time.sleep(5)
    settings = {'B_CHECKPOINT': str(target), 'B_EXECUTE_HORIZON': '5', 'B_DENOISING_STEPS': '4'}
    os.environ.update(settings)
    status('training_scene_diagnostic', seed=1)
    os.environ.update(B_START_SEED='1', B_EPISODES='1', B_TAG='recovery_B_h5_d4_train')
    run(['bash', 'scripts/evaluate_rgb_B.sh'], 'rgb-B-training-scene.log')
    status('independent_20_seed_model_evaluation', start_seed=200000)
    os.environ.update(B_START_SEED='200000', B_EPISODES='20', B_TAG='recovery_B_h5_d4_test20')
    run(['bash', 'scripts/evaluate_rgb_B.sh'], 'rgb-B-independent-test20.log')
    records = json.loads((root / 'evaluation/gr00t_lift_pot_recovery_B_h5_d4_test20.json').read_text())
    assert len(records) == 20 and len({r['seed'] for r in records}) == 20
    assert all(not r['expert_called'] and r['checkpoint'] == str(target) for r in records)
    # Separate setup/inference aborts from completed model rollouts.
    executed = [r for r in records if r['actions'] > 0]
    successful = [r for r in executed if r['success']]
    aborted = [r for r in records if r['actions'] == 0]
    summary = {'checkpoint': str(target), 'configuration': settings, 'requested_seeds': 20,
               'executed_seeds': len(executed), 'successes': len(successful),
               'failures': len(executed)-len(successful), 'initialization_or_inference_aborts': len(aborted),
               'success_rate_executed': len(successful)/len(executed) if executed else None,
               'training_scene_excluded': True, 'success_criterion': 'RoboTwin lift_pot.check_success',
               'records': 'evaluation/gr00t_lift_pot_recovery_B_h5_d4_test20.json'}
    if successful:
        raw = Path(successful[0]['video'])
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(raw), '-f', 'null', '-'], check=True)
        video = root / 'videos/01_lift_pot_gr00t.mp4'
        assert not video.exists(), 'Preserving existing success video'
        shutil.copyfile(raw, video)
        summary['success_video'] = str(video)
        summary['success_video_source_seed'] = successful[0]['seed']
        summary['raw_success_video'] = str(raw)
    (root / 'results/rgb_B_closed_loop_summary.json').write_text(json.dumps(summary, indent=2))
    note = '\n\n## B 后台实测结果（自动记录，待分析）\n\n```json\n' + json.dumps(summary, ensure_ascii=False, indent=2) + '\n```\n'
    for filename in ['results/gr00t_recovery_test.md', 'experiment_report.md']:
        with (root / filename).open('a') as stream:
            stream.write(note)
    status('completed_prepared_B_comparison', **summary)
except Exception:
    error = traceback.format_exc()
    status('failed_preserving_all_artifacts', error=error)
    print(error, flush=True)
    raise
finally:
    if server is not None and server.poll() is None:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
