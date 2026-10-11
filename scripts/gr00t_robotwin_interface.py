"""Observation/action boundary for a future verified ALOHA GR00T checkpoint.

This module does not load an expert, replay a trajectory, or claim policy success.
GR00T decode_action already converts relative joints to absolute targets.
"""
import numpy as np

KEYS = ("left_arm", "left_gripper", "right_arm", "right_gripper")
CAMERAS = {"head": "head_camera", "left_wrist": "left_camera", "right_wrist": "right_camera"}

def observation_for_policy(observation, instruction):
    joints = observation["joint_action"]
    images = {}
    shape = None
    for key, source in CAMERAS.items():
        frame = np.asarray(observation["observation"][source]["rgb"])
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"{key}: expected uint8 HWC RGB; implicit color/range conversion forbidden")
        if shape is not None and frame.shape != shape:
            raise ValueError("Three camera image shapes must agree")
        shape = frame.shape
        images[key] = frame[None, None]
    for key, size in zip(KEYS, (6, 1, 6, 1)):
        state = np.asarray(joints[key])
        if state.size != size or not np.isfinite(state).all():
            raise ValueError(f"{key}: invalid state size or nonfinite values")
    return {
        "video": images,
        "state": {key: np.asarray(joints[key], dtype=np.float32).reshape(1, 1, -1)
                  for key in KEYS},
        "language": {"annotation.human.task_description": [[instruction]]},
    }

def action_chunk_for_robot(actions, joint_limits):
    parts = [np.asarray(actions[key], dtype=np.float32) for key in KEYS]
    assert all(p.ndim == 3 and p.shape[0] == 1 for p in parts)
    assert [p.shape[-1] for p in parts] == [6, 1, 6, 1]
    assert len({p.shape[1] for p in parts}) == 1
    chunk = np.concatenate(parts, axis=-1)[0]
    if not np.isfinite(chunk).all():
        raise ValueError("Nonfinite GR00T action")
    limits = np.asarray(joint_limits, dtype=np.float32)
    assert limits.shape == (12, 2)
    joints = chunk[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]
    if np.any(joints < limits[:, 0]) or np.any(joints > limits[:, 1]):
        raise ValueError("GR00T action outside joint limits")
    if np.any(chunk[:, [6, 13]] < 0) or np.any(chunk[:, [6, 13]] > 1):
        raise ValueError("GR00T normalized gripper target outside [0,1]")
    return chunk

def infer_chunk(policy, observation, instruction, joint_limits):
    actions, info = policy.get_action(observation_for_policy(observation, instruction))
    return action_chunk_for_robot(actions, joint_limits), info

# Decoded absolute targets feed the project fixed-period executor.
# Physical tracking remains a separate measured quantity.
