"""Offline audit helper checks; no saved run results or neural forwards."""
import unittest
import json
from itertools import product
import numpy as np

from research_program.triadic_partner_ecology_study import audit_execution as audit
from research_program.triadic_partner_ecology_study import design


def values_for(states, actions):
    states = np.asarray(states, dtype=np.int16)
    actions = np.asarray(actions, dtype=np.int16)
    reward, executed, satisfied = audit.previous.native_settlement(states, actions)
    return dict(states=states, action_indices=actions, greedy_reward=reward,
        executed=executed, satisfied=satisfied,
        conditional_exact_expected_reward=np.zeros(len(states)),
        conditional_exact_full_success_probability=np.zeros(len(states)),
        conditional_exact_execution_probability=np.zeros(len(states)))


class AuditExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs, cls.layers, cls.excluded = audit.independent_specs()

    def test_independent_support_and_monitor_match_frozen_design(self):
        prepared = json.loads(json.dumps(design.make_prepared()))
        self.assertEqual(self.specs, prepared['partitions'])
        self.assertEqual(self.layers, prepared['common_destination_layers'])
        self.assertEqual(self.excluded, prepared['excluded_destination_layers'])
        self.assertEqual([len(self.specs[e]['train']['needs']) for e in audit.ECOLOGIES], [324, 996])

    def test_weights_have_equal_strata_not_equal_worlds(self):
        for e, p in product(audit.ECOLOGIES, audit.PARTITIONS):
            with self.subTest(ecology=e, partition=p):
                spec = self.specs[e][p]
                full = audit.weights_for(spec)
                self.assertTrue(np.array_equal(full, design.evaluation_weights(spec)))
                self.assertAlmostEqual(float(full.sum()), 1., places=14)
                physical = len(spec['layouts'])*6
                for layer in spec['demand_strata']:
                    ids = np.concatenate([np.arange(n*physical, (n+1)*physical) for n in layer['need_indices']])
                    self.assertAlmostEqual(float(full[ids].sum()), 1/21, places=14)
                self.assertGreater(len(np.unique(full)), 1)
                self.assertTrue(np.array_equal(audit.weights_for(spec, True), np.full(672, 1/672)))

    def test_sampler_pairing_and_both_private_owners(self):
        rng = np.random.default_rng(7131)
        u = rng.random((256, 4))
        u[0] = 0
        u[1] = np.nextafter(1., 0.)
        for p in audit.PARTITIONS:
            draws = {}
            for e in audit.ECOLOGIES:
                spec = self.specs[e][p]
                ids, d, li, oi = audit.sample(spec, u)
                self.assertTrue(np.array_equal(ids, design.sample_indices(spec, u)))
                original = design.pairing_fields(spec, ids)
                for name, value in zip(('destinations', 'layout_indices', 'owner_indices'), (d, li, oi)):
                    self.assertTrue(np.array_equal(value, original[name]))
                draws[e] = (ids, d, li, oi)
            for k in (1, 2, 3):
                self.assertTrue(np.array_equal(draws['unique'][k], draws['multiple'][k]))
            self.assertFalse(np.array_equal(draws['unique'][0], draws['multiple'][0]))

    def test_role_topology_physics_are_distinct(self):
        state = [0, 0, 3, 0, 1, 2, 3, 1, 2, 3]
        actions = [[1, 1, 0], [2, 0, 1], [1, 5, 0], [1, 1, 1], [3, 3, 0]]
        weights = np.array([.1, .2, .3, .15, .25])
        result, raw = audit.weighted_measures(values_for([state]*5, actions), weights)
        for name, expected in dict(compatible_role_rate=.65, topology_consistency_rate=.85,
                                  physical_match_rate=.55, full_success_rate=.1, reward_mean=.2).items():
            self.assertAlmostEqual(result[name], expected)
        self.assertEqual(result['fixed_pair_oracles']['AB']['gamma'], 1.)
        self.assertEqual(result['fixed_pair_oracles']['AC']['gamma'], 0.)
        self.assertEqual(raw['full_success_worlds'], 1)
        self.assertEqual(raw['observed_joint_actions'], 5)
        self.assertEqual(result['unique_edge_subgroups']['AB']['population_weight'], 1.)
        self.assertIsNone(result['unique_edge_subgroups']['AC']['compatible_role_rate'])

    def test_full_distribution_gamma_and_monitor_scope(self):
        for ecology, gamma in (('unique', 1/3), ('multiple', 31/39)):
            spec = self.specs[ecology]['train']
            states = np.asarray([need+[0, 1, 2, 3, 1, 2, 3] for need in spec['needs']], dtype=np.int16)
            physical = len(spec['layouts'])*6
            weights = audit.weights_for(spec).reshape(len(states), physical).sum(1)
            actions = np.tile([1, 1, 0], (len(states), 1))
            out, _ = audit.weighted_measures(values_for(states, actions), weights)
            for pair in audit.PAIR_NAMES:
                self.assertAlmostEqual(out['fixed_pair_oracles'][pair]['gamma'], gamma, places=14)
            self.assertAlmostEqual(out['compatible_role_rate'], gamma, places=14)
        # A monitor subset concentrated on AB need tables has its own gamma.
        states = [[0, 0, 3, 0, 1, 2, 3, 1, 2, 3]]
        out, _ = audit.weighted_measures(values_for(states, [[1, 1, 0]]), np.array([1.]))
        self.assertEqual(out['best_fixed_pair_gamma'], 1.)

    def test_reject_bad_weights_or_reward(self):
        values = values_for([[0, 0, 3, 0, 1, 2, 3, 1, 2, 3]], [[1, 1, 0]])
        for weights in (np.array([.99]), np.array([-1.]), np.array([np.nan])):
            with self.assertRaises(AssertionError): audit.weighted_measures(values, weights)
        values['greedy_reward'][0] = .5
        with self.assertRaises(AssertionError): audit.weighted_measures(values, np.array([1.]))

    def test_primary_uses_four_seeds_and_unique_minus_multiple(self):
        rows = []
        for si, seed in enumerate(audit.SEEDS):
            for ecology, condition in product(audit.ECOLOGIES, audit.CONDITIONS):
                effect = (.20+.01*si if ecology == 'unique' else .08+.02*si)
                role = .25+(effect if condition == 'PI_live' else .03 if condition == 'FI_live' else 0)
                w = dict(compatible_role_rate=role, topology_consistency_rate=.8, physical_match_rate=.7,
                    full_success_rate=role/2, reward_mean=.4, best_fixed_pair_gamma=.5,
                    compatible_role_excess_best_fixed_pair=role-.5)
                rows.append(dict(seed=seed, ecology=ecology, condition=condition,
                    final={'heldout_layouts': {'natural': w}}))
        out = audit.primary_from_summaries(rows)
        self.assertEqual(len(out['seed_pairs']), 4)
        for si, row in enumerate(out['seed_pairs']):
            self.assertAlmostEqual(row['primary_DiD_unique_minus_multiple'], .12-.01*si)
            self.assertAlmostEqual(row['FI_DiD_unique_minus_multiple'], 0.)
        self.assertAlmostEqual(out['equal_seed_means']['primary_DiD_unique_minus_multiple'], .105)
        self.assertAlmostEqual(out['equal_seed_means']['full_success_DiD_unique_minus_multiple'], .0525)

    def test_helper_source_anchors_and_closed_route(self):
        for module, relative in ((audit.previous,
            'research_program/triadic_message_study/results/messages_001/audit_execution_001/verification.json'),
            (audit.previous.pure, 'research_program/triadic_learning_baseline/results/learning_001/audit_execution_001/verification.json')):
            self.assertEqual(audit.sha(module.__file__), audit.read(audit.ROOT/relative)['audit_source_sha256'])
        tokens = np.arange(12).reshape(1, 3, 4) % 8
        live = audit.previous.route(tokens, True)
        closed = audit.previous.route(tokens, False)
        for viewer in range(3):
            self.assertTrue(np.array_equal(live[:, viewer, 32*viewer:32*(viewer+1)],
                                           closed[:, viewer, 32*viewer:32*(viewer+1)]))
            self.assertEqual(closed[0, viewer, 96:].tolist(), [float(i == viewer) for i in range(3)])
            for other in set(range(3))-{viewer}:
                self.assertFalse(closed[:, viewer, 32*other:32*(other+1)].any())


if __name__ == '__main__': unittest.main()
