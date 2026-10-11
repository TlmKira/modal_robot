"""CPU-only URDF FK audit; never executes dataset actions in the simulator."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

root = Path(__file__).resolve().parents[1]
robot_dir = root / 'simulation/RoboTwin/assets/embodiments/aloha-agilex'
config = yaml.safe_load((robot_dir/'config.yml').read_text())
tree = ET.parse(robot_dir/config['urdf_path'])
joints = {j.find('child').attrib['link']: j for j in tree.findall('joint')}
origin = config['robot_pose'][0]
world = np.eye(4)
world[:3,3] = origin[:3]
world[:3,:3] = Rotation.from_quat([origin[4],origin[5],origin[6],origin[3]]).as_matrix()

def fk(link, q):
    if link not in joints:
        return world
    joint=joints[link]
    parent=fk(joint.find('parent').attrib['link'],q)
    transform=np.eye(4)
    pos=joint.find('origin')
    if pos is not None:
        transform[:3,3]=np.fromstring(pos.get('xyz','0 0 0'),sep=' ')
        transform[:3,:3]=Rotation.from_euler('xyz',np.fromstring(pos.get('rpy','0 0 0'),sep=' ')).as_matrix()
    motion=np.eye(4)
    if joint.get('type') in ('revolute','continuous'):
        axis=np.fromstring(joint.find('axis').get('xyz'),sep=' ')
        motion[:3,:3]=Rotation.from_rotvec(axis*q.get(joint.get('name'),0)).as_matrix()
    return parent@transform@motion


def poses(values):
    output=[]
    for side,offset,names in [('left',0,config['arm_joints_name'][0]),('right',7,config['arm_joints_name'][1])]:
        link=next(j.find('child').get('link') for j in tree.findall('joint') if j.get('name')==names[-1])
        pose=fk(link,dict(zip(names,values[offset:offset+6])))
        pose[:3,:3] = pose[:3,:3] @ np.asarray(config['global_trans_matrix']) @ np.asarray(config['delta_matrix'])
        pose[:3,3] += pose[:3,:3] @ np.array([config['gripper_bias'],0,0])
        output.append(pose)
    return output
