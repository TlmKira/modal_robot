"""Compare original recorded camera geometry and states with a same-seed live scene."""
import json,os,sys
from pathlib import Path
import numpy as np,cv2,h5py,yaml
root=Path(__file__).resolve().parents[1];repo=root/'simulation/RoboTwin';os.chdir(repo);sys.path.insert(0,str(repo))
from envs.lift_pot import lift_pot
from data.decode_image_bit import decode_image_bit
cfg=yaml.safe_load((repo/'env_cfg/task_config/demo_clean.yml').read_text());rf=repo/'assets/embodiments/aloha-agilex';rc=yaml.safe_load((rf/'config.yml').read_text())
cfg.update(task_name='lift_pot',task_config='demo_clean',embodiment_name='aloha-agilex',left_robot_file=str(rf),right_robot_file=str(rf),left_embodiment_config=rc,right_embodiment_config=rc,dual_arm_embodied=True,need_plan=True,save_data=False,save_path=str(root/'datasets/local_diagnostic'),render_freq=0,save_freq=15,eval_mode=True,eval_video_save_dir=None)
task=lift_pot();task.setup_demo(seed=1,now_ep_num=1,**cfg);obs=task.get_obs();result={'seed':1,'model_id':int(task.model_id),'pot_position':task.pot.get_pose().p.tolist(),'live_state':obs['joint_action']['vector'].tolist(),'cameras':{}}
with h5py.File(root/'datasets/lift_pot/aloha-agilex_clean_50/data/episode1.hdf5') as f:
 result['source_state']=f['joint_action/vector'][0].tolist()
 for key in ['head_camera','left_camera','right_camera']:
  source=decode_image_bit(f[f'observation/{key}/rgb'][0]); live=obs['observation'][key]['rgb']; cam={'source_shape':list(source.shape),'live_shape':list(live.shape),'source_mean':source.mean(axis=(0,1)).tolist(),'live_mean':live.mean(axis=(0,1)).tolist()}
  for geom in ['intrinsic_cv','extrinsic_cv','cam2world_gl']:
   original=f[f'observation/{key}/{geom}'][0]; current=obs['observation'][key][geom]; cam[geom]={'source':original.tolist(),'live':current.tolist(),'max_difference':float(np.max(np.abs(original-current)))}
  result['cameras'][key]=cam; cv2.imwrite(str(root/f'results/screenshots/recovery_source_{key}.png'),source[...,::-1]); cv2.imwrite(str(root/f'results/screenshots/recovery_live_{key}.png'),live[...,::-1])
(root/'evaluation/live_source_scene_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);task.close_env()
