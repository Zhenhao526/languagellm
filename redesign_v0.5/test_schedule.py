import unittest
import numpy as np
import torch
from camp import ImageBank, make_agents, remake_agents, projected_banks
from run_stages import A, training_schedule, training_batch_plan, rollout


class ScheduleTests(unittest.TestCase):
    def test_mixed_preserves_batch_and_exploration_multiset(self):
        levels, order = training_schedule(24001, 1800, 'course')
        mixed, perm = training_schedule(24001, 1800, 'mixed')
        self.assertTrue(np.array_equal(mixed, levels[perm]))
        self.assertTrue(np.array_equal(np.sort(perm), order))
        self.assertTrue(np.array_equal(perm[1500:], order[1500:]))
        self.assertEqual([int((levels == k).sum()) for k in (2, 3, 4)], [400, 400, 1000])
        self.assertFalse(np.array_equal(perm[:1500], order[:1500]))
        early = set()
        for j in range(100):
            early.update(training_batch_plan(24001, j, 2, A['hidden_sequence'])['allowed_sites'])
        self.assertEqual(early, {0, 1, 2, 3})

    def test_public_access_constraint_and_same_batch_identity(self):
        torch.set_num_threads(1)
        bank = ImageBank()
        base = make_agents(765)
        agents = remake_agents(765, [a.state_dict() for a in base])
        banks = projected_banks(agents, bank)
        plan = training_batch_plan(765, 10, 2, A['hidden_sequence'])
        stats, _, traces = rollout(agents, banks, bank, plan, 1701, 64,
                                  training=True, greedy=False, trace=True, split='train')
        for row in traces:
            self.assertTrue(np.isin(row['place'], plan['allowed_sites']).all())
            self.assertTrue(np.isin(row['positions'], plan['allowed_sites']).all())
        clone = dict(plan, known=True)
        stats2, _, _ = rollout(agents, banks, bank, clone, 1701, 64,
                              training=True, greedy=False, split='train')
        self.assertEqual(stats['world_sha256'], stats2['world_sha256'])


if __name__ == '__main__':
    unittest.main()
