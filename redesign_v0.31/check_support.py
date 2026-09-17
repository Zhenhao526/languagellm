"""Synthetic checks for the v0.31 population schedule and fixtures."""
import unittest,itertools
import numpy as np
import support

class Tests(unittest.TestCase):
    def table(self):
        maps=np.repeat(np.arange(30),12);photos=np.tile(np.asarray([[i,j] for i in range(6) for j in range(2)],np.int64),(30,1));positions=support.MAPS[maps];return dict(map_id=np.r_[maps,maps],photo_ids=np.tile(photos,(2,1)),positions=np.tile(positions,(2,1)),shown=np.repeat(np.arange(2),360))
    def test_design_and_population(self):
        self.assertEqual(support.DESIGN_SHA256,'df69de1f994802ec6256af4b62e23d87c96663affc54bf343a8768d961c286da');self.assertEqual(support.CONDITIONS,('fixed_partners','rotating_partners'));self.assertEqual(support.matching('fixed_partners',7),((0,1),(2,3)));self.assertEqual(support.matching('rotating_partners',0),((0,1),(2,3)));self.assertEqual(support.matching('rotating_partners',1),((0,3),(2,1)))
    def test_graph_invariants(self):
        for p in (1,2,3):
            g=support.groups(p,'fixed_partners');self.assertEqual(len(g['train12']),12);self.assertEqual(len(g['target12']),12);self.assertEqual(set(g['train12'])&set(g['target12']),set());self.assertTrue(np.all(np.bincount(support.MAPS[g['train12'],0],minlength=6)==2));self.assertTrue(np.all(np.bincount(support.MAPS[g['train12'],1],minlength=6)==2));self.assertTrue(np.all(np.bincount(support.MAPS[g['target12'],0],minlength=6)==2));self.assertTrue(np.all(np.bincount(support.MAPS[g['target12'],1],minlength=6)==2))
    def test_fixture_balanced_and_condition_paired(self):
        w=self.table();a=support.fixture(34101,1,0,0,'fixed_partners',w);b=support.fixture(34101,1,0,0,'rotating_partners',w);np.testing.assert_array_equal(a['indices'],b['indices']);np.testing.assert_array_equal(a['uniforms'],b['uniforms']);idx=a['indices'];self.assertEqual(len(idx),240);self.assertTrue(np.all(np.bincount(w['map_id'][idx],minlength=30)[support.groups(1,'fixed_partners')['train12']]==20));self.assertTrue(np.all(np.bincount(w['positions'][idx,0],minlength=6)==40));self.assertTrue(np.all(np.bincount(w['positions'][idx,1],minlength=6)==40));before=np.random.get_state();support.fixture(34101,1,0,0,'fixed_partners',w);after=np.random.get_state();self.assertEqual(before[0],after[0]);np.testing.assert_array_equal(before[1],after[1]);self.assertEqual(before[2:],after[2:])
    def test_invalid(self):
        w=self.table();
        with self.assertRaises(ValueError):support.fixture(1,1,2,0,'fixed_partners',w)
        with self.assertRaises(ValueError):support.fixture(1,1,0,0,'bad',w)
        with self.assertRaises(ValueError):support.matching('rotating_partners',-1)

if __name__=='__main__':unittest.main(verbosity=2)
