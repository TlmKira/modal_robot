"""Real RGB/state -> trained GR00T -> official RoboTwin qpos execution. No expert."""
import argparse
import json
from multiprocessing.connection import Client
import os
from pathlib import Path
import sys
import time
import traceback
import imageio.v2 as imageio
import numpy as np
import sapien
import yaml
from recovery_control import execute_fixed
from gr00t_robotwin_interface import observation_for_policy, action_chunk_for_robot

parser = argparse.ArgumentParser()
parser.add_argument('--start-seed', type=int, default=200000)
parser.add_argument('--episodes', type=int, default=3)
parser.add_argument('--max-actions', type=int, default=300)
parser.add_argument('--execute-horizon', type=int, default=10)
parser.add_argument('--measured-control-start', action='store_true')
parser.add_argument('--measured-state', action='store_true')
parser.add_argument('--velocity-limit', type=float)
parser.add_argument('--denoising-steps', type=int)
parser.add_argument('--ensemble', type=int)
parser.add_argument('--checkpoint', default='checkpoints/lift_pot_2000/checkpoint-2000')
parser.add_argument('--port', type=int, default=6007)
parser.add_argument('--tag', default='topp_h10')
parser.add_argument('--control', choices=['topp','fixed'], default='topp')
parser.add_argument('--physics-steps-per-action', type=int, default=15)
parser.add_argument('--match-legacy-training-channels', action='store_true')
opts = parser.parse_args()
if opts.physics_steps_per_action <= 0 or opts.execute_horizon <= 0 or opts.execute_horizon > 40:
    parser.error('Positive physics ticks and execution horizon 1..40 required')
if opts.velocity_limit is not None and opts.velocity_limit <= 0:
    parser.error('Velocity limit must be positive')
root = Path(__file__).resolve().parents[1]
repo = root / 'simulation/RoboTwin'
os.chdir(repo)
sys.path.insert(0, str(repo))
from envs.lift_pot import lift_pot

config = yaml.safe_load((repo / 'env_cfg/task_config/demo_clean.yml').read_text())
robot_file = repo / 'assets/embodiments/aloha-agilex'
robot_config = yaml.safe_load((robot_file / 'config.yml').read_text())
config.update(task_name='lift_pot', task_config='demo_clean', embodiment_name='aloha-agilex',
              left_robot_file=str(robot_file), right_robot_file=str(robot_file),
              left_embodiment_config=robot_config, right_embodiment_config=robot_config,
              dual_arm_embodied=True, need_plan=True, save_data=False,
              save_path=str(root / 'datasets/local_policy'), render_freq=0, save_freq=8,
              eval_mode=True, eval_video_save_dir=None)
conn = Client(('127.0.0.1', opts.port), authkey=b'local-gr00t-simulation')
results = []
for seed in range(opts.start_seed, opts.start_seed + opts.episodes):
    prefix = f'lift_pot_gr00t_{opts.tag}_seed{seed}'
    raw = root / f'videos/{prefix}_raw.mp4'
    if raw.exists():
        raise FileExistsError(f'Preserving existing video: {raw}')
    task = lift_pot()
    writer = imageio.get_writer(raw, fps=250/8, codec='libx264', quality=8, macro_block_size=1)
    item = dict(seed=seed, checkpoint=str((root / opts.checkpoint).resolve()),
                policy_source='live Gr00tPolicy.get_action', expert_called=False, success=False,
                measured_state=opts.measured_state, measured_control_start=opts.measured_control_start, velocity_limit=opts.velocity_limit,
                execute_horizon=opts.execute_horizon, control=opts.control,
                physics_steps_per_action=opts.physics_steps_per_action,
                model_input_red_blue_swapped=opts.match_legacy_training_channels,
                video=str(raw), tag=opts.tag)
    frames, physics_steps = [0], [0]
    predicted, executed, observed, calls = [], [], [], []
    policy_states = []
    try:
        task.setup_demo(seed=seed, now_ep_num=seed, **config)
        task.step_lim = opts.max_actions
        joints = task.robot.left_arm_joints + task.robot.right_arm_joints
        limits = np.asarray([joint.get_limits()[0] for joint in joints])
        position = np.array([1.2, -1.6, 1.55])
        forward = np.array([0, -0.25, 0.78]) - position
        forward /= np.linalg.norm(forward)
        left = np.cross([0, 0, 1], forward)
        left /= np.linalg.norm(left)
        matrix = np.eye(4)
        matrix[:3, :3] = np.stack([forward, left, np.cross(forward,left)], axis=1)
        matrix[:3, 3] = position
        camera = task.scene.add_camera('presentation', 960, 720, np.deg2rad(52), 0.1, 10)
        camera.entity.set_pose(sapien.Pose(matrix))
        def capture():
            task._update_render()
            camera.take_picture()
            rgb = np.clip(camera.get_picture('Color')[..., :3]*255, 0,255).astype(np.uint8)
            writer.append_data(rgb)
            frames[0] += 1
        original_step = task.scene.step
        def recorded_step():
            original_step()
            physics_steps[0] += 1
            if physics_steps[0] % 8 == 0:
                capture()
        task.scene.step = recorded_step
        capture()
        conn.send({'op':'reset', 'seed':seed})
        assert conn.recv()['ok']
        item['initial_pot_position'] = task.pot.get_pose().p.tolist()
        item['model_id'] = int(task.model_id)
        instruction = 'Lift the pot with both arms.'
        while task.take_action_cnt < task.step_lim and not task.eval_success:
            observation = task.get_obs()
            observed.append(np.asarray(observation['joint_action']['vector']))
            if opts.measured_state:
                q = np.asarray(task.robot.get_left_arm_real_jointState()+task.robot.get_right_arm_real_jointState())
                for key,sl in [('left_arm',slice(0,6)),('left_gripper',slice(6,7)),('right_arm',slice(7,13)),('right_gripper',slice(13,14))]:
                    observation['joint_action'][key] = q[sl]
            model_observation = observation_for_policy(observation, instruction)
            policy_states.append(np.concatenate([model_observation['state'][key][0,0] for key in ('left_arm','left_gripper','right_arm','right_gripper')]))
            if opts.match_legacy_training_channels:
                model_observation['video'] = {key: np.ascontiguousarray(value[..., ::-1])
                                              for key,value in model_observation['video'].items()}
            request = {'op':'action', 'observation':model_observation}
            if opts.denoising_steps is not None:
                request['denoising_steps'] = opts.denoising_steps
            if opts.ensemble is not None:
                request['ensemble'] = opts.ensemble
            conn.send(request)
            response = conn.recv()
            if 'error' in response:
                raise RuntimeError(response['error'])
            if Path(response['checkpoint']).resolve() != Path(item['checkpoint']):
                raise RuntimeError('Server checkpoint does not match requested evaluation checkpoint')
            item['denoising_steps'] = response.get('denoising_steps',4)
            item['sampling_ensemble'] = response.get('ensemble',1)
            predicted_chunk = np.concatenate([response['actions'][key][0] for key in
                ('left_arm','left_gripper','right_arm','right_gripper')],axis=-1)
            predicted.append(predicted_chunk)
            chunk = action_chunk_for_robot(response['actions'],limits)
            real_joints = np.asarray(task.robot.get_left_arm_real_jointState() + task.robot.get_right_arm_real_jointState())
            left_distance = float(np.linalg.norm(np.asarray(task.robot.get_left_tcp_pose()[:3]) - np.asarray(task.pot.get_contact_point(0)[:3])))
            right_distance = float(np.linalg.norm(np.asarray(task.robot.get_right_tcp_pose()[:3]) - np.asarray(task.pot.get_contact_point(1)[:3])))
            calls.append(dict(inference_seconds=response['inference_seconds'], action_count=task.take_action_cnt,
                              pot_height=float(task.pot.get_pose().p[2]),
                              real_joints=real_joints.tolist(),
                              max_joint_tracking_error=float(np.max(np.abs(real_joints-observed[-1]))),
                              left_grasp_distance=left_distance, right_grasp_distance=right_distance))
            for action in chunk[:opts.execute_horizon]:
                if opts.control == 'topp':
                    task.take_action(action, action_type='qpos')
                else:
                    execute_fixed(task, action, opts.physics_steps_per_action, opts.velocity_limit, opts.measured_control_start)
                executed.append(np.asarray(task.robot.get_left_arm_jointState()+task.robot.get_right_arm_jointState()))
                if task.eval_success or task.take_action_cnt >= task.step_lim:
                    break
            print(json.dumps({'seed':seed, 'actions':task.take_action_cnt,
                              'pot_height':float(task.pot.get_pose().p[2]), 'success':bool(task.eval_success),
                              'left_grasp_distance':left_distance, 'right_grasp_distance':right_distance}),flush=True)
            if float(task.pot.get_pose().p[2]) < item['initial_pot_position'][2] - 0.12:
                raise RuntimeError('Pot fell below table; stopping failed rollout')
        capture()
        item.update(success=bool(task.check_success()), actions=task.take_action_cnt,
                    final_pot_position=task.pot.get_pose().p.tolist(),
                    final_pot_quaternion=task.pot.get_pose().q.tolist(),
                    final_left_tcp=np.asarray(task.robot.get_left_tcp_pose()[:3]).tolist(),
                    final_right_tcp=np.asarray(task.robot.get_right_tcp_pose()[:3]).tolist(),
                    final_left_grasp=np.asarray(task.pot.get_contact_point(0)[:3]).tolist(),
                    final_right_grasp=np.asarray(task.pot.get_contact_point(1)[:3]).tolist(), inference_calls=calls)
    except (Exception, KeyboardInterrupt) as exc:
        traceback.print_exc()
        item['error'] = repr(exc)
        if hasattr(task, 'pot'):
            item['final_pot_position'] = task.pot.get_pose().p.tolist()
        if isinstance(exc, KeyboardInterrupt):
            raise
    finally:
        writer.close()
        item['inference_calls'] = calls
        item['actions'] = getattr(task, 'take_action_cnt', len(executed))
        item.update(frames=frames[0], physics_steps=physics_steps[0], fps=250/8,
                    duration_seconds=frames[0]/(250/8))
        np.savez(root / f'evaluation/{prefix}_actions.npz', predictions=np.asarray(predicted),
                 executed=np.asarray(executed), states=np.asarray(observed), policy_states=np.asarray(policy_states))
        if hasattr(task, 'scene'):
            task.close_env()
        results.append(item)
        (root / f'evaluation/gr00t_lift_pot_{opts.tag}.json').write_text(json.dumps(results,indent=2))
        print(json.dumps({k:v for k,v in item.items() if k != 'inference_calls'}),flush=True)
    if item['success']:
        (root / 'evaluation/gr00t_success_video_source.txt').write_text(str(raw))
conn.close()
