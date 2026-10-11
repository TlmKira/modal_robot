"""Decode completed model rollouts; never rename a failure to the success deliverable."""
import json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1];checks=[]
for p in sorted((root/'evaluation').glob('gr00t_lift_pot_recovery*.json')):
 for r in json.loads(p.read_text()):
  if not r.get('completed_rollout',True) or not r.get('valid_factor_comparison',True):continue
  video=r['video']; probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate:format=duration','-of','json',video])); decode=subprocess.run(['ffmpeg','-v','error','-i',video,'-f','null','-'],capture_output=True,text=True)
  checks.append({'tag':r['tag'],'seed':r['seed'],'official_success':r['success'],'video':video,'probe':probe,'decode_ok':decode.returncode==0 and not decode.stderr,'decode_error':decode.stderr});assert checks[-1]['decode_ok']
(root/'evaluation/recovery_video_checks.json').write_text(json.dumps(checks,indent=2));print('decoded',len(checks))
