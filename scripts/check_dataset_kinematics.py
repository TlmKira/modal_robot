"""CPU-only URDF FK audit; never executes dataset actions in the simulator."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import h5py
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

errors={'left':[],'right':[]}
for path in sorted((root/'datasets/lift_pot/aloha-agilex_clean_50/data').glob('*.hdf5')):
    with h5py.File(path) as source:
        values=source['joint_action/vector'][:]
        for index in [0,len(values)//2,len(values)-1]:
            for side, offset, names in [('left',0,config['arm_joints_name'][0]),('right',7,config['arm_joints_name'][1])]:
                q=dict(zip(names,values[index,offset:offset+6]))
                end_link=next(j.find('child').get('link') for j in tree.findall('joint') if j.get('name')==names[-1])
                position=fk(end_link,q)[:3,3]
                recorded=source[f'endpose/{side}_endpose'][index,:3]
                errors[side].append(float(np.linalg.norm(position-recorded)))
result=dict(source='CPU-only FK at 3 frames per episode; no trajectory replay',
            caveat='Dataset joints are commanded targets while endposes measure physics, so small tracking differences are expected.',
            errors_m={side:dict(median=float(np.median(x)),max=float(np.max(x)),samples=len(x)) for side,x in errors.items()})
(root/'evaluation/dataset_kinematics_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
