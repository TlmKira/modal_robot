"""Report completed real policy rollouts and verify their videos."""
import csv
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
groups = ['fixed60ms_h10','fixed60ms_h40_channelmatch','fixed60ms_h10_channelmatch_d16_e4']
trials = []
media = []
for group in groups:
    path = root / f'evaluation/gr00t_lift_pot_{group}.json'
    if not path.exists():
        continue
    for trial in json.loads(path.read_text()):
        if trial.get('frames',0) == 0 or 'KeyboardInterrupt' in trial.get('error',''):
            continue
        assert not trial['expert_called']
        trials.append(trial)
        probe = subprocess.run(['ffprobe','-v','error','-show_entries',
            'format=duration:stream=codec_name,width,height,avg_frame_rate,nb_frames',
            '-of','json',trial['video']],capture_output=True,text=True,check=True)
        decode = subprocess.run(['ffmpeg','-v','error','-i',trial['video'],'-f','null','-'],
                                capture_output=True,text=True)
        media.append(dict(video=trial['video'],probe=json.loads(probe.stdout),
                          full_decode_returncode=decode.returncode,decode_errors=decode.stderr))
metrics = root / 'results/metrics.csv'
with metrics.open() as stream:
    reader = csv.DictReader(stream)
    columns = reader.fieldnames
    old = list(reader)
for trial in trials:
    policy = f'gr00t_n17_2000_{trial["tag"]}'
    if any(row['policy']==policy and row['seed']==str(trial['seed']) for row in old):
        continue
    old.append(dict(task='lift_pot',policy=policy,view='wide',seed=trial['seed'],
                    success=trial['success'],pot_height_m=trial['final_pot_position'][2],
                    frames=trial['frames'],video=trial['video']))
with metrics.open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(old)
summary = dict(completed_rollouts=len(trials),unique_seeds=len({t['seed'] for t in trials}),
               successful_rollouts=sum(t['success'] for t in trials),
               checkpoint=str(root/'checkpoints/lift_pot_2000/checkpoint-2000'),
               groups={group:dict(trials=sum(t['tag']==group for t in trials),
                                  successes=sum(t['success'] for t in trials if t['tag']==group))
                       for group in groups},
               limitation='Small diagnostic sample; controller/input/sampling configurations differ. Repeat seeds are not independent trials.')
(root/'evaluation/gr00t_video_checks.json').write_text(json.dumps(media,indent=2))
(root/'results/gr00t_closed_loop_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
