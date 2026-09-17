"""Pure synthetic/native-enumeration checks; no actor initialization or training."""
from copy import deepcopy
from itertools import permutations, product
import unittest
import numpy as np
from research_program.triadic_action_dependency_study import environment as env
from research_program.triadic_action_dependency_study import dataset, metrics as m


def onehot(actions):
    return np.eye(17, dtype=np.float64)[np.asarray(actions)]


def native_truth(state):
    plans = env.full_success_plans(state.needs, state.layout)
    assert len(plans) == 1
    i, j, site, dest = plans[0]
    result = [0, 0, 0]
    for who, partner in ((i, j), (j, i)):
        result[who] = env.all_actions(env.AGENTS[who]).index(dict(kind='transport',
            site=env.SITES[site], destination=env.DESTINATIONS[dest], partner=env.AGENTS[partner]))
    return result


def state_row(state):
    return list(state.needs + state.layout + state.private_sites)


class WorldTests(unittest.TestCase):
    def test_actual_layout_truth_matches_environment_across_all_permutations(self):
        # Four distinct semantic/role plans, all 24 resource arrangements.
        seen = {}
        for need in env.support():
            p = env.full_success_plans(need)[0]
            seen.setdefault(p, need)
            if len(seen) == 4:
                break
        states = [env.State(n, layout, (3, 1, 2)) for n in seen.values()
                  for layout in permutations(range(4))]
        packed = np.asarray([state_row(s) for s in states])
        actual = m.truth_actions(packed)
        expected = np.asarray([native_truth(s) for s in states])
        np.testing.assert_array_equal(actual, expected)
        report = m.world_metrics(states=packed, action_indices=actual,
                                 action_probabilities=onehot(actual), greedy_reward=np.ones(len(states)))
        self.assertEqual(report['full_success_rate'], 1)
        self.assertEqual(report['role_site_destination_success_rate'], 1)
        self.assertEqual(report['expected_reward_given_saved_messages'], 1)
        self.assertEqual(report['full_probability_given_saved_messages'], 1)
        # Applying the canonical site's index unchanged is observably wrong.
        wrong = np.asarray([native_truth(env.State(s.needs, (0, 1, 2, 3))) for s in states])
        self.assertGreater(np.count_nonzero(np.any(wrong != actual, axis=1)), 0)
        self.assertLess(m.world_metrics(states=packed, action_indices=wrong)['full_success_rate'], 1)

    def test_failures_and_components_do_not_condition_on_success(self):
        s = env.State(env.support()[0], (3, 2, 1, 0))
        target = native_truth(s)
        i, j, site, dest = env.full_success_plans(s.needs, s.layout)[0]
        third = 3 - i - j
        rows = [target.copy() for _ in range(6)]
        rows[1][i] = 0  # solo: one required actor waits
        rows[2][third] = 1  # overload
        rows[3][i] = 1 + ((site + 1) % 4) * 4 + (target[i] - 1) % 4
        rows[4][i] = 1 + site * 4 + (1 - dest) * 2 + (target[i] - 1) % 2
        rows[5][i] = 1 + site * 4 + dest * 2 + (1 - (target[i] - 1) % 2)
        actions = np.asarray(rows)
        packed = np.asarray([state_row(s)] * len(rows))
        saved = []
        for a in actions:
            native = env.settle(s, {g: env.all_actions(g)[int(k)] for g, k in zip(env.AGENTS, a)})
            saved.append(native['reward'])
        report = m.world_metrics(states=packed, action_indices=actions, greedy_reward=saved)
        self.assertEqual(report['full_success_worlds'], 1)
        self.assertEqual(report['failure_worlds'], 5)
        self.assertEqual(report['reward_counts']['0.0'], 5)
        self.assertEqual(report['actual_active_agent_count']['1'], 1)
        self.assertEqual(report['actual_active_agent_count']['3'], 1)
        self.assertEqual(report['physical_execution_rate'], 1 / 6)
        # Site/destination can be correct when stated partners are wrong.
        self.assertGreater(report['site_destination_success_rate'], report['full_success_rate'])
        self.assertGreater(report['role_success_rate'], report['full_success_rate'])
        with self.assertRaisesRegex(ValueError, 'saved reward'):
            m.world_metrics(states=packed, action_indices=actions, greedy_reward=np.ones(6))

    def test_exact_probabilities_against_all_4913_native_joint_actions(self):
        s = env.State(env.support()[17], (2, 0, 3, 1))
        p = np.random.default_rng(8173).dirichlet(np.arange(1, 18), size=3)
        choices = np.argmax(p, -1)[None, :]
        result = m.world_metrics(states=np.asarray([state_row(s)]),
            action_indices=choices, action_probabilities=p[None])
        expected = full = physical = 0.
        for actions in product(range(17), repeat=3):
            q = float(np.prod([p[a, k] for a, k in enumerate(actions)]))
            settled = env.settle(s, {g: env.all_actions(g)[k] for g, k in zip(env.AGENTS, actions)})
            expected += q * settled['reward']
            full += q * (settled['reward'] == 1)
            physical += q * any(r['executed'] for r in settled['individual_feedback'].values())
        self.assertAlmostEqual(result['expected_reward_given_saved_messages'], expected, places=14)
        self.assertAlmostEqual(result['full_probability_given_saved_messages'], full, places=14)
        self.assertAlmostEqual(result['physical_probability_given_saved_messages'], physical, places=14)

    def test_probability_first_argmax_and_nonunique_truth_rejected(self):
        s = env.State(env.support()[0], (0, 1, 2, 3))
        packed = np.asarray([state_row(s)])
        p = np.full((1, 3, 17), 1 / 17)
        m.world_metrics(states=packed, action_indices=np.zeros((1, 3), dtype=int), action_probabilities=p)
        with self.assertRaisesRegex(ValueError, 'first argmax'):
            m.world_metrics(states=packed, action_indices=np.ones((1, 3), dtype=int), action_probabilities=p)
        with self.assertRaisesRegex(ValueError, 'normalized'):
            m.world_metrics(states=packed, action_indices=np.zeros((1, 3), dtype=int), action_probabilities=p / 2)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            m.truth_actions(np.asarray([[0, 0, 0, 0, 1, 2, 3, 1, 2, 3]]))


class SemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = dataset.make_prepared()['partitions']['new_needs_and_layouts']
        nl, no = len(cls.spec['layouts']), len(cls.spec['private_sites'])
        cls.ids = (np.arange(len(cls.spec['needs']))[:, None] * nl * no + np.array([0, no + 1])).ravel()
        cls.states = dataset.pack_states(cls.spec, cls.ids)
        cls.truth = m.truth_actions(cls.states)
        cls.pairs = dataset.content_pairs(cls.spec)

    def kwargs(self, actions=None):
        choices = self.truth if actions is None else actions
        return dict(spec=self.spec, states=self.states, state_indices=self.ids,
                    action_indices=choices, action_probabilities=onehot(choices), pairs=self.pairs)

    def test_oracle_all_cases_and_two_background_scope(self):
        before = deepcopy(self.pairs)
        result = m.semantic_metrics(**self.kwargs(), include_rows=True)
        self.assertEqual(result['coverage'], 'all_needs_fixed_background_subset')
        self.assertEqual(result['saved_background_count'], 2)
        self.assertEqual(result['case_count'], len(self.pairs['rows']))
        self.assertEqual(result['case_background_pairs'], 2 * len(self.pairs['rows']))
        for group in ('content', 'role'):
            self.assertEqual(result[group]['macro']['both_endpoints_apt'], 1)
            self.assertEqual(result[group]['macro']['both_team_full_success'], 1)
        self.assertEqual(result['content']['macro']['both_endpoint_correct_probability_product'], 1)
        self.assertEqual(before, self.pairs)
        for row, value in zip(result['case_rows'], result['case_mean_values']['both_endpoints_apt']):
            self.assertEqual(value, 1)
            self.assertIn(row['classification'], ('content', 'role', 'descriptive'))

    def test_silent_same_local_observation_has_structural_zero(self):
        # A deterministic local policy: own need + public material + assigned
        # private material. No other current need enters the policy.
        actions = np.empty((len(self.states), 3), dtype=np.int16)
        for who in range(3):
            private_material = self.states[np.arange(len(self.states)), 3 + self.states[:, 7 + who]]
            actions[:, who] = (self.states[:, who] + self.states[:, 3] + private_material) % 17
        result = m.semantic_metrics(**self.kwargs(actions))
        self.assertEqual(result['content']['macro']['both_endpoints_apt'], 0)
        self.assertEqual(result['role']['macro']['both_endpoints_apt'], 0)
        self.assertEqual(result['content']['macro']['listener_action_changed'], 0)
        self.assertEqual(result['content']['case_count'], sum(r['classification'] == 'content' for r in self.pairs['rows']))

    def test_equal_axes_and_sender_listener_not_raw_case_weight(self):
        rows = self.pairs['rows']
        v = np.asarray([float(r['axis_index'] == 2) for r in rows])
        result = m._case_aggregate({'probe': v}, rows, 'content')
        self.assertAlmostEqual(result['macro']['probe'], 1 / 3)
        content = np.asarray([r['classification'] == 'content' for r in rows])
        self.assertNotAlmostEqual(v[content].mean(), 1 / 3)
        z = np.asarray([float(r['sender'] == 0 and r['listener'] == 1) for r in rows])
        self.assertAlmostEqual(m._case_aggregate({'probe': z}, rows, 'content')['macro']['probe'], 1 / 6)
        bad = deepcopy(rows)
        next(r for r in bad if r['classification'] == 'content')['within_axis_case_weight_denominator'] += 1
        with self.assertRaisesRegex(ValueError, 'weights'):
            m._case_aggregate({'probe': v}, bad, 'content')

    def test_missing_cases_backgrounds_or_mispacked_layout_refused(self):
        kw = self.kwargs()
        kw['pairs'] = {'rows': self.pairs['rows'][:-1]}
        with self.assertRaisesRegex(ValueError, 'predetermined'):
            m.semantic_metrics(**kw)
        kw = self.kwargs()
        for key in ('states', 'state_indices', 'action_indices', 'action_probabilities'):
            kw[key] = kw[key][1:]
        with self.assertRaisesRegex(ValueError, 'same saved backgrounds'):
            m.semantic_metrics(**kw)
        kw = self.kwargs(); kw['states'] = self.states.copy()
        kw['states'][0, [3, 4]] = kw['states'][0, [4, 3]]
        with self.assertRaisesRegex(ValueError, 'Saved states'):
            m.semantic_metrics(**kw)


def enriched_fixture():
    result = []
    for k, seed in enumerate(m.SEEDS):
        content = {'FI_silent': .2, 'FI_live': .3, 'PL_silent': 0., 'LL_silent': 0.,
                   'PL_live': [.3, .15, .2, .4][k], 'LL_live': .2}
        full = {'FI_silent': .4, 'FI_live': .7, 'PL_silent': .2,
                'PL_live': .6 + .05*k, 'LL_silent': .1, 'LL_live': .5}
        for condition in m.CONDITIONS:
            cell = dict(semantic_metrics=dict(coverage='full_partition',
                content=dict(macro=dict(both_endpoints_apt=content[condition]))),
                world_metrics={name: full[condition] for name in m.COMPARISON_METRICS
                               if name != 'content_both_endpoints_apt'})
            closed = deepcopy(cell)
            if condition.endswith('_live'):
                closed['semantic_metrics']['content']['macro']['both_endpoints_apt'] = 0.
                closed['world_metrics'] = {name: v / 2 for name, v in closed['world_metrics'].items()}
            result.append(dict(seed=seed, condition=condition, updates=6000,
                final={part: dict(natural=deepcopy(cell), closed=deepcopy(closed)) for part in m.PARTITIONS}))
    return result


class ComparisonTests(unittest.TestCase):
    def test_all_paired_seeds_negative_preserved_and_contrasts_distinct(self):
        inputs = enriched_fixture(); before = deepcopy(inputs)
        out = m.primary_comparison(inputs)
        np.testing.assert_allclose([r['difference'] for r in out['primary']['seed_values']], [.1, -.05, 0, .2], atol=1e-15)
        self.assertAlmostEqual(out['primary']['equal_seed_mean'], .0625)
        self.assertAlmostEqual(out['secondary_task_difference_in_differences']['equal_seed_mean'], .075)
        self.assertLess(out['primary']['minimum'], 0)
        self.assertEqual(len(out['partitions']), 4)
        self.assertEqual(inputs, before)

    def test_incomplete_or_monitor_endpoint_or_silent_violation_refused(self):
        with self.assertRaisesRegex(ValueError, '24'):
            m.primary_comparison(enriched_fixture()[:-1])
        x = enriched_fixture()
        x[0]['final']['train']['natural']['semantic_metrics']['coverage'] = 'all_needs_fixed_background_subset'
        with self.assertRaisesRegex(ValueError, 'monitor'):
            m.primary_comparison(x)
        x = enriched_fixture()
        next(r for r in x if r['condition'] == 'PL_silent')['final']['train']['natural']['semantic_metrics']['content']['macro']['both_endpoints_apt'] = .1
        with self.assertRaisesRegex(ValueError, 'structural zero'):
            m.primary_comparison(x)
        x = enriched_fixture(); x[0]['updates'] = 3000
        with self.assertRaisesRegex(ValueError, '6000'):
            m.primary_comparison(x)


if __name__ == '__main__':
    unittest.main()
