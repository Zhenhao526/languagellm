import unittest
import numpy as np
from research_program.triadic_formation_trajectory_study import metrics as m

class TrajectoryTests(unittest.TestCase):
    def test_area_constant_linear_and_signed(self):
        self.assertEqual(m.normalized_area([2]*6),2)
        self.assertEqual(m.normalized_area(np.asarray(m.STEPS)/6000),.5)
        self.assertEqual(m.normalized_area(-np.asarray(m.STEPS)/6000),-.5)
        # Equal checkpoint weighting is wrong for this nonuniform time grid.
        self.assertEqual(m.normalized_area([0,0,0,0,0,1]),.25)
        with self.assertRaises(ValueError):m.normalized_area([0]*5)
        with self.assertRaises(ValueError):m.normalized_area([0]*6,[0,100,500,1500,3000,5999])

    def test_receiver_row_reference_and_axis_orientation(self):
        keys=('diagonal_effect','offdiagonal_effect','selectivity','uniform_position_effect','diagonal_minus_uniform_position')
        points={t:dict(current={k:t/6000 for k in keys},retrospective_final={k:-t/6000 for k in keys}) for t in m.STEPS}
        cross={(r,s):{'contrast':{'macro':{'target_apt':i*.1+j*.01}}} for i,r in enumerate(m.STEPS) for j,s in enumerate(m.STEPS)}
        out=m.trajectory(points,cross);adjusted=np.asarray(out['cross_time_minus_receiver_same_time'])
        np.testing.assert_array_equal(np.diag(adjusted),0)
        np.testing.assert_allclose(adjusted[5],np.arange(6)*.01-.05,atol=1e-15)
        self.assertEqual(out['normalized_areas']['selectivity'],.5)
        self.assertEqual(out['retrospective_normalized_areas']['selectivity'],-.5)
        del cross[(0,100)]
        with self.assertRaises(ValueError):m.trajectory(points,cross)

    def test_exact_message_persistence_and_slot_turnover_separate(self):
        a=np.zeros((419904,2,3,4),np.int8);b=a.copy();b[:,1,2,3]=1
        p={'selected_positions':np.zeros((3,3),int).tolist()};q={'selected_positions':np.ones((3,3),int).tolist()}
        out=m.temporal_change(a,b,p,q)
        self.assertAlmostEqual(out['train_token_change_rate'],1/24)
        self.assertEqual(out['sender_whole_packet_change_rate'],[0,0,1])
        self.assertEqual(out['selected_position_change_rate'],1)
        # Code can stay constant even when a selection tie convention differs.
        self.assertEqual(m.temporal_change(a,a,p,q)['train_token_change_rate'],0)

if __name__=='__main__':unittest.main()
