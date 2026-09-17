"""Synthetic checks for v0.32 schedules, paired fixtures and joint payoff."""
import unittest
import numpy as np
import support, joint


class Tests(unittest.TestCase):
    def table(self):
        maps = np.repeat(np.arange(30), 12)
        photos = np.tile(np.asarray([[i, j] for i in range(6) for j in range(2)], np.int64), (30, 1))
        positions = support.MAPS[maps]
        return dict(map_id=np.r_[maps, maps], photo_ids=np.tile(photos, (2, 1)), positions=np.tile(positions, (2, 1)), shown=np.repeat(np.arange(2), 360))

    def test_design_and_schedule(self):
        self.assertEqual(support.DESIGN_SHA256, '2c388c4cc804c99505d3af83c40af75d916ead60bc999e480427a74830cecf9c')
        self.assertEqual(support.CONDITIONS, ('fixed_partners', 'rotating_partners'))
        self.assertEqual(support.matching('fixed_partners', 7), ((0, 1), (2, 3)))
        self.assertEqual(support.matching('rotating_partners', 0), ((0, 1), (2, 3)))
        self.assertEqual(support.matching('rotating_partners', 1), ((0, 3), (2, 1)))

    def test_graph_invariants(self):
        for panel in (1, 2, 3):
            groups = support.groups(panel, 'fixed_partners')
            self.assertEqual(len(groups['train12']), 12); self.assertEqual(len(groups['target12']), 12)
            self.assertEqual(set(groups['train12']) & set(groups['target12']), set())
            for key in ('train12', 'target12'):
                self.assertTrue(np.all(np.bincount(support.MAPS[groups[key], 0], minlength=6) == 2))
                self.assertTrue(np.all(np.bincount(support.MAPS[groups[key], 1], minlength=6) == 2))

    def test_fixture_balanced_and_paired(self):
        worlds = self.table(); fixed = support.fixture(34101, 1, 0, 0, 'fixed_partners', worlds); rotating = support.fixture(34101, 1, 0, 0, 'rotating_partners', worlds)
        np.testing.assert_array_equal(fixed['indices'], rotating['indices']); np.testing.assert_array_equal(fixed['uniforms'], rotating['uniforms'])
        self.assertEqual(fixed['uniforms'].shape, (240, 8)); idx = fixed['indices']
        counts = np.bincount(worlds['map_id'][idx], minlength=30); self.assertTrue(np.all(counts[support.groups(1, 'fixed_partners')['train12']] == 20))
        self.assertTrue(np.all(np.bincount(worlds['positions'][idx, 0], minlength=6) == 40)); self.assertTrue(np.all(np.bincount(worlds['positions'][idx, 1], minlength=6) == 40))
        before = np.random.get_state(); support.fixture(34101, 1, 0, 0, 'fixed_partners', worlds); after = np.random.get_state(); self.assertEqual(before[0], after[0]); np.testing.assert_array_equal(before[1], after[1]); self.assertEqual(before[2:], after[2:])

    def test_joint_reward(self):
        zeros = np.zeros((1, 2), np.float32); one = np.ones((1, 2), np.float32); one_zero = np.asarray([[1, 0]], np.float32)
        self.assertEqual(float(joint.joint_reward(zeros, zeros)[0]), 0.)
        self.assertEqual(float(joint.joint_reward(one_zero, zeros)[0]), .125)
        self.assertEqual(float(joint.joint_reward(one, one)[0]), 1.)

    def test_invalid(self):
        worlds = self.table()
        with self.assertRaises(ValueError): support.fixture(1, 1, 2, 0, 'fixed_partners', worlds)
        with self.assertRaises(ValueError): support.fixture(1, 1, 0, 0, 'bad', worlds)
        with self.assertRaises(ValueError): support.matching('rotating_partners', -1)
        with self.assertRaises(ValueError): joint.joint_reward(np.zeros((2, 3), np.float32), np.zeros((2, 3), np.float32))


if __name__ == '__main__': unittest.main(verbosity=2)
