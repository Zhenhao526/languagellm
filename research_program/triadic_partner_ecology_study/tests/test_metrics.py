from copy import deepcopy
from fractions import Fraction
from itertools import product
import math
import unittest

import numpy as np

from research_program.triadic_partner_ecology_study import metrics as m
from research_program.triadic_task import environment as env


def packed(needs=(0, 0, 0)):
    return list(needs)+[0, 1, 2, 3, 1, 2, 3]


class WeightedMetricTests(unittest.TestCase):
    def test_all_1728_need_graphs_agree_with_original(self):
        needs = np.array(list(product(range(12), repeat=3)))
        actual = m.compatibility(needs)
        for n, row in zip(needs, actual):
            edges = env.compatible_pairs(tuple(n))
            self.assertEqual(row.tolist(), [pair in edges for pair in m.PAIRS])

    def test_all_51_action_decodings(self):
        for agent in range(3):
            acts = np.zeros((17, 3), dtype=np.int64)
            acts[:, agent] = np.arange(17)
            dec = m.decode_actions(acts)
            for index, action in enumerate(env.all_actions(env.AGENTS[agent])):
                if index == 0:
                    self.assertEqual((dec['site'][index, agent], dec['destination'][index, agent], dec['partner'][index, agent]), (-1, -1, -1))
                else:
                    self.assertEqual((dec['site'][index, agent], dec['destination'][index, agent], dec['partner'][index, agent]),
                        (env.SITES.index(action['site']), env.DESTINATIONS.index(action['destination']), env.AGENTS.index(action['partner'])))

    def test_role_does_not_require_physical_agreement(self):
        states = np.array([packed()]*2)
        actions = np.array([[1, 5, 0], [1, 3, 0]])
        r = m.summarize(states, actions, np.zeros(2), np.array([.7, .3]))['weighted']
        self.assertEqual(r['compatible_role_rate'], 1.)
        self.assertEqual(r['topology_consistency_rate'], 1.)
        self.assertEqual(r['physical_match_rate'], 0.)
        self.assertEqual(r['full_success_rate'], 0.)
        self.assertAlmostEqual(r['site_mismatch_given_topology'], .7)
        self.assertAlmostEqual(r['destination_mismatch_given_topology'], .3)

    def test_physical_pair_can_be_incompatible_and_half_reward(self):
        r = m.summarize(np.array([packed((0, 3, 0))]), np.array([[1, 1, 0]]), np.array([.5]), np.array([1.]))['weighted']
        self.assertEqual(r['compatible_role_rate'], 0.)
        self.assertEqual(r['topology_consistency_rate'], 1.)
        self.assertEqual(r['physical_match_rate'], 1.)
        self.assertEqual(r['reward_mean'], .5)
        self.assertEqual(r['unique_edge_subgroups']['AC']['compatible_role_rate'], 0.)
        self.assertEqual(r['unique_edge_subgroups']['AC']['isolated_wait_rate'], 0.)

    def test_weights_not_raw_world_counts_and_sparse_frequency(self):
        states = np.array([packed(), packed()])
        result = m.summarize(states, np.array([[1, 1, 0], [0, 0, 0]]), np.array([1., 0.]), np.array([.8, .2]))
        self.assertAlmostEqual(result['weighted']['full_success_rate'], .8)
        self.assertEqual(result['raw']['full_success_worlds'], 1)
        self.assertEqual(result['raw']['observed_joint_actions'], 2)
        self.assertAlmostEqual(sum(r['probability_weight'] for r in result['raw']['joint_action_distribution']), 1.)
        self.assertEqual(sum(r['raw_worlds'] for r in result['raw']['joint_action_distribution']), 2)

    def test_full_candidate_oracles_use_layer_weights(self):
        candidates = []
        for d in product(range(3), repeat=3):
            groups = {1: [], 2: []}
            for r in product(range(4), repeat=3):
                n = tuple(3*r[a]+d[a] for a in range(3))
                k = len(env.compatible_pairs(n))
                if k == 1: groups[1].append(n)
                elif k >= 2: groups[2].append(n)
            if all(groups.values()): candidates.append(groups)
        self.assertEqual(len(candidates), 21)
        for ecology, expected in ((1, Fraction(1, 3)), (2, Fraction(31, 39))):
            needs, weights = [], []
            for group in candidates:
                needs.extend(group[ecology])
                weights.extend([float(Fraction(1, 21*len(group[ecology])))]*len(group[ecology]))
            n = len(needs)
            result = m.summarize(np.array([packed(x) for x in needs]), np.zeros((n, 3), dtype=int), np.zeros(n), np.array(weights))['weighted']
            for oracle in result['fixed_pair_oracles'].values():
                self.assertAlmostEqual(oracle['gamma'], float(expected), places=12)
                self.assertAlmostEqual(oracle['mean_reward_oracle'], float((1+expected)/2), places=12)
                self.assertAlmostEqual(oracle['mean_log_J_oracle'], -float(1-expected)*math.log(2), places=12)
            if ecology == 1:
                for subgroup in result['unique_edge_subgroups'].values():
                    self.assertAlmostEqual(subgroup['population_weight'], 1/3)
            else:
                for subgroup in result['unique_edge_subgroups'].values():
                    self.assertEqual(subgroup['population_weight'], 0.)
                    self.assertIsNone(subgroup['compatible_role_rate'])

    def test_all_4913_joint_actions_original_score_comparison(self):
        state = env.State((0, 3, 0), (0, 1, 2, 3))
        actions = np.array(list(product(range(17), repeat=3)))
        reward = np.array([env.settle(state, {a: env.all_actions(a)[row[i]] for i, a in enumerate(env.AGENTS)}, require_match=True)['reward'] for row in actions])
        result = m.summarize(np.tile(packed(state.needs), (len(actions), 1)), actions, reward,
            np.full(len(actions), 1/len(actions)))
        self.assertEqual(result['raw']['observed_joint_actions'], 4913)
        self.assertAlmostEqual(result['weighted']['topology_consistency_rate'], 192/4913)
        # All partner-consistent site/destination choices: 3pairs×8×8=192 topologies.
        self.assertAlmostEqual(result['weighted']['physical_match_rate'], 24/4913)

    def test_invalid_weight_action_and_reward_rejected(self):
        states = np.array([packed()]); acts = np.array([[1, 1, 0]])
        for weights in (np.array([.9]), np.array([float('nan')]), np.array([-1.])):
            with self.assertRaises(ValueError): m.summarize(states, acts, np.array([1.]), weights)
        with self.assertRaises(ValueError): m.summarize(states, acts.astype(float), np.array([1.]), np.ones(1))
        with self.assertRaises(ValueError): m.summarize(states, acts, np.array([0.]), np.ones(1))

    def test_primary_preserves_all32_and_did_sign(self):
        records = []
        for seed, ecology, condition in product(m.SEEDS, m.ECOLOGIES, m.CONDITIONS):
            role = (.2 if ecology == 'unique' else .4)
            if condition == 'PI_live': role += .3 if ecology == 'unique' else .1
            if condition == 'FI_live': role += .05
            weighted = dict(compatible_role_rate=role, topology_consistency_rate=.9,
                physical_match_rate=.7, full_success_rate=.2, reward_mean=.3,
                best_fixed_pair_gamma=1/3, compatible_role_excess_best_fixed_pair=role-1/3)
            records.append(dict(seed=seed, ecology=ecology, condition=condition,
                final={'heldout_layouts': {'natural': {'weighted': weighted}}}))
        result = m.primary_comparison(records)
        self.assertEqual(len(result['seed_pairs']), 4)
        self.assertAlmostEqual(result['equal_seed_means']['primary_DiD_unique_minus_multiple'], .2)
        self.assertAlmostEqual(result['equal_seed_means']['FI_DiD_unique_minus_multiple'], 0.)
        self.assertEqual(sum(len(v) for v in result['seed_pairs'][0]['eight_cells'].values()), 8)
        with self.assertRaises(ValueError): m.primary_comparison(records[:-1])
        with self.assertRaises(ValueError): m.primary_comparison(records+[deepcopy(records[0])])


if __name__ == '__main__':
    unittest.main()
