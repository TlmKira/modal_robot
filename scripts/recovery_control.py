"""Shared 60 ms drive executor used by model and diagnostic expert replay."""
import numpy as np

def execute_fixed(task, action, count=15, velocity_limit=None, measured_start=False):
    current = np.array(task.robot.get_left_arm_real_jointState() + task.robot.get_right_arm_real_jointState()) if measured_start else np.array(task.robot.get_left_arm_jointState() + task.robot.get_right_arm_jointState())
    if velocity_limit is not None:
        action = np.asarray(action).copy()
        indices = [0,1,2,3,4,5,7,8,9,10,11,12]
        maximum_delta = velocity_limit * count * task.scene.get_timestep()
        action[indices] = current[indices] + np.clip(action[indices]-current[indices],-maximum_delta,maximum_delta)
    velocity = (action-current)/(count*task.scene.get_timestep())
    task.take_action_cnt += 1
    for index in range(1,count+1):
        target = current + (action-current)*index/count
        task.robot.set_arm_joints(target[:6], velocity[:6], 'left')
        task.robot.set_arm_joints(target[7:13], velocity[7:13], 'right')
        task.robot.set_gripper(float(target[6]), 'left')
        task.robot.set_gripper(float(target[13]), 'right')
        task.scene.step()
        if task.check_success():
            task.eval_success = True
            break

