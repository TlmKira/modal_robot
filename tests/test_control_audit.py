"""CPU-only regression checks: no model or SAPIEN imports."""
import ast,json,sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from control_audit import validate_action_timing,summarize_hold,SuccessEvents
from gr00t_robotwin_interface import observation_for_policy,action_chunk_for_robot

class AuditTests(unittest.TestCase):
    def test_timing_rejects_scaled_control(self):
        self.assertAlmostEqual(validate_action_timing(.004),.06)
        for dt in [.002,.008,float('nan'),0]:
            with self.assertRaises(ValueError):validate_action_timing(dt)

    def test_hold_excludes_initial_and_overshoot(self):
        samples=[dict(tick=t,official=True,both_contact=True) for t in range(10,525)]
        h=summarize_hold(10,525,samples)
        self.assertTrue(h['continuous_official']);self.assertEqual(h['both_contact_seconds'],2.)
        samples[200]['official']=False
        self.assertFalse(summarize_hold(10,525,samples)['continuous_official'])
        self.assertFalse(summarize_hold(10,509,samples)['completed'])
        self.assertFalse(summarize_hold(10,525,samples[2:])['completed'])
        self.assertFalse(summarize_hold(None,525,samples)['completed'])

    def test_events_at_exact_ticks(self):
        e=SuccessEvents()
        self.assertEqual(e.observe(1,False),[])
        self.assertEqual(e.observe(2,True),['first_official_success'])
        self.assertEqual(e.observe(3,True),[])
        self.assertEqual(e.observe(4,False),['official_success_lost'])
        self.assertEqual(e.observe(5,True),['official_success_regained'])
        self.assertEqual(e.first_success_tick,2)

    def test_rgb_and_action_boundaries(self):
        obs={'observation':{k:{'rgb':np.full((2,3,3),[v,2,3],np.uint8)} for k,v in zip(['head_camera','left_camera','right_camera'],[11,22,33])},'joint_action':{k:np.zeros(n) for k,n in zip(['left_arm','left_gripper','right_arm','right_gripper'],[6,1,6,1])}}
        p=observation_for_policy(obs,'Lift')
        self.assertEqual([int(p['video'][k][0,0,0,0,0]) for k in ['head','left_wrist','right_wrist']],[11,22,33])
        obs['observation']['head_camera']['rgb']=np.zeros((2,3,3),np.float32)
        with self.assertRaises(ValueError):observation_for_policy(obs,'Lift')
        acts={k:np.full((1,40,n),v,np.float32) for k,n,v in zip(['left_arm','left_gripper','right_arm','right_gripper'],[6,1,6,1],[.1,.2,.3,.4])}
        chunk=action_chunk_for_robot(acts,np.tile([-10,10],(12,1)))
        np.testing.assert_allclose(chunk[0],[.1]*6+[.2]+[.3]*6+[.4])

    def test_executor_preserves_baseline_targets(self):
        def executor(file):
            tree=ast.parse(file.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
            ns={'np':np,'validate_action_timing':validate_action_timing};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(file),'exec'),ns);return ns['execute']
        class Robot:
            def __init__(self):self.q=np.zeros(14);self.calls=[]
            def get_left_arm_jointState(self):return self.q[:7].tolist()
            def get_right_arm_jointState(self):return self.q[7:].tolist()
            def set_arm_joints(self,q,v,s):self.q[slice(0,6) if s=='left' else slice(7,13)]=q;self.calls.append((s,np.array(q),np.array(v)))
            def set_gripper(self,q,s):self.q[6 if s=='left' else 13]=q
        results=[]
        for file in [ROOT/'docs/baseline_2026-10-10/BC_blind_runner.py',ROOT/'scripts/BC_blind_runner.py']:
            robot=Robot();rec=SimpleNamespace(tick=0,contacts=lambda:{},record_event=lambda *args:None)
            def step():rec.tick+=1
            task=SimpleNamespace(robot=robot,scene=SimpleNamespace(step=step,get_timestep=lambda:.004),take_action_cnt=0,check_success=lambda:rec.tick>=8)
            executor(file)(task,np.array([.1]*6+[.2]+[.3]*6+[.4]),rec,{'first_tick':None,'hold_ticks':[],'events':SuccessEvents()})
            self.assertEqual(rec.tick,15);results.append(robot.calls)
        for a,b in zip(*results):np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[2],b[2])

    def test_saved_results_unchanged_if_available(self):
        folder=ROOT/'evaluation/BC_blind_v2'
        if not folder.exists():self.skipTest('Historical raw JSON kept outside Git; copy artifacts to run this check')
        counts={m:[0,0] for m in ['B','C']}
        for m in counts:
            paths=list(folder.glob(m+'_seed*.json'));self.assertEqual(len(paths),20)
            for p in paths:
                r=json.loads(p.read_text());h=summarize_hold(r['first_success_tick'],r['physics_ticks'],r['hold_telemetry'])
                self.assertEqual(h['continuous_official'],r['hold_2s_continuous_official']);counts[m][0]+=r['official_success'];counts[m][1]+=h['continuous_official']
        self.assertEqual(counts,{'B':[0,0],'C':[9,6]})

if __name__=='__main__':unittest.main()
