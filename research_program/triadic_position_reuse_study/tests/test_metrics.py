"""Pure arrays/static task tests. No policy files, inference, or preparation."""
from copy import deepcopy
from itertools import permutations
import unittest
import numpy as np

from research_program.triadic_position_reuse_study import dataset as d
from research_program.triadic_position_reuse_study import metrics as m


def position_reports(spec, effect):
    """Continuous synthetic scores permit exact additive-contrast checks."""
    reports = {}
    n = len(spec['sender'])
    for unit in range(8):
        same = {'target_apt': np.full((2, n), .35),
                'natural_current_apt': np.ones((2, n)),
                'packet_in_train_endpoint_greedy_set': np.zeros((2, n), bool)}
        opposite = {k: v.copy() for k, v in same.items()}
        opposite['target_apt'] += np.broadcast_to(effect(unit), (2, n))
        differences = [m.contrast({k: v[b] for k, v in same.items()},
                                  {k: v[b] for k, v in opposite.items()}) for b in (0, 1)]
        reports[unit] = dict(same=same, opposite=opposite,
                             contrast={k: np.stack([x[k] for x in differences]) for k in differences[0]})
    return reports


class MetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepared = d.make_prepared(d.load_sources())
        cls.spec = prepared['validation']['arrays']
        ps = prepared['validation']['metadata']['source_part_spec']
        cls.states = np.stack([d.task.pack_states(ps, cls.spec['endpoint_indices'][:, b]) for b in (0, 1)], axis=1)

    def fixture(self, direction=0, actions=None):
        n = len(self.spec['axis'])
        states = self.states[:, direction].copy()
        counterpart = self.states[:, 1-direction].copy()
        if actions is None:
            actions = self.spec['correct_actions'][:, 1-direction].copy()
        natural = {'action_indices': self.spec['correct_actions'][:, direction].copy(),
                   'messages': np.zeros((n, 2, 3, 4), np.int8)}
        data = dict(m.old_channel.settle(states, actions))
        data.update(action_indices=actions.copy(), action_probabilities=np.eye(17)[actions],
                    messages=np.zeros((n, 2, 3, 4), np.int8),
                    patched_outward_packets=np.zeros((n, 2, 4), np.int8),
                    counterfactual_reward=m.old_channel.settle(counterpart, actions)['greedy_reward'])
        code_sets = [np.array([0], np.uint32) for _ in range(3)]
        return states, counterpart, natural, data, code_sets

    def compute(self, fixture, direction=0):
        states, counterpart, natural, data, code_sets = fixture
        return m.row_values(self.spec, direction, states, natural, data, code_sets,
                            counterfactual_states=counterpart)

    def test_all_actions_decode_actual_layout_and_wait(self):
        # Independent decode through the public action menu and material tuples.
        packed = self.states[0, 0]
        rows, actions, listeners, expected = [], [], [], []
        for layout in permutations(range(4)):
            for actor in range(3):
                for index, action in enumerate(d.env.all_actions(d.env.AGENTS[actor])):
                    state = packed.copy(); state[3:7] = layout
                    acts = np.zeros(3, np.int8); acts[actor] = index
                    rows.append(state); actions.append(acts); listeners.append(actor)
                    if index == 0:
                        expected.append((-1, -1, -1, -1))
                    else:
                        material = layout[d.env.SITES.index(action['site'])]
                        kind, length = d.env.MATERIALS[material]
                        expected.append((('wood', 'fiber').index(kind), ('short', 'long').index(length),
                                         d.env.DESTINATIONS.index(action['destination']),
                                         d.env.AGENTS.index(action['partner'])))
        actual = m.decoded_attributes(np.asarray(rows), np.asarray(actions), np.asarray(listeners))
        np.testing.assert_array_equal(actual, expected)

    def test_decode_rejects_invalid_actions_states_and_listeners(self):
        states, _, _, data, _ = self.fixture()
        valid = data['action_indices']
        for actions in (valid.astype(float), np.full_like(valid, 17), np.full_like(valid, -1)):
            with self.subTest(kind='action'), self.assertRaises(ValueError):
                m.decoded_attributes(states, actions, self.spec['listener'])
        bad = states.copy(); bad[:, 3:7] = 0
        with self.assertRaises(ValueError):
            m.decoded_attributes(bad, valid, self.spec['listener'])
        with self.assertRaises(ValueError):
            m.decoded_attributes(states, valid, np.full(144, 3))

    def test_explicit_counterfactual_truth_and_input_immutability_both_directions(self):
        self.assertNotIn('counterpart_states', self.spec)
        for b in (0, 1):
            fixture = self.fixture(b)
            before = deepcopy(fixture)
            result = self.compute(fixture, b)
            for key in ('target_apt', 'counterfactual_team_full', 'physical_pair_executed',
                        'unchanged_requirements_apt', 'kind_apt', 'length_apt', 'destination_apt',
                        'partner_apt', 'packet_in_train_endpoint_greedy_set'):
                self.assertTrue(result[key].all(), key)
            self.assertFalse(result['current_apt'].any())
            self.assertTrue(result['natural_current_apt'].all())
            for old, new in zip(before, fixture):
                if isinstance(old, dict):
                    for key in old:
                        np.testing.assert_array_equal(old[key], new[key])
                elif isinstance(old, list):
                    for left, right in zip(old, new):
                        np.testing.assert_array_equal(left, right)
                else:
                    np.testing.assert_array_equal(old, new)

    def test_counterfactual_is_required_and_only_sender_need_may_change(self):
        f = self.fixture(); states, counterpart, natural, data, codes = f
        with self.assertRaises(TypeError):
            m.row_values(self.spec, 0, states, natural, data, codes)
        for which in ('layout', 'owner', 'other_need', 'unchanged_sender'):
            cf = counterpart.copy()
            if which == 'layout':
                cf[:, [3, 4]] = cf[:, [4, 3]]
            elif which == 'owner':
                cf[:, [7, 8]] = cf[:, [8, 7]]
            elif which == 'other_need':
                r = np.arange(144); listener = self.spec['listener']
                cf[r, listener] = (cf[r, listener] + 1) % 24
            else:
                cf[:, :3] = states[:, :3]
            with self.subTest(which=which), self.assertRaises(ValueError):
                m.row_values(self.spec, 0, states, natural, data, codes, counterfactual_states=cf)

    def test_current_action_preserves_unchanged_fields_but_wait_does_not(self):
        current = self.spec['correct_actions'][:, 0]
        result = self.compute(self.fixture(actions=current))
        self.assertTrue(result['current_apt'].all())
        self.assertFalse(result['target_apt'].any())
        self.assertTrue(result['unchanged_requirements_apt'].all())
        for i, attribute in enumerate(m.ATTRIBUTES):
            expected = self.spec['axis'] != i
            np.testing.assert_array_equal(result[attribute + '_apt'], expected)
        waiting = self.compute(self.fixture(actions=np.zeros((144, 3), np.int8)))
        for key in ('current_apt', 'target_apt', 'unchanged_requirements_apt', 'physical_pair_executed',
                    'kind_apt', 'length_apt', 'destination_apt', 'partner_apt'):
            self.assertFalse(waiting[key].any(), key)

    def test_wrong_unchanged_partner_rejects_attribute_conjunction(self):
        actions = self.spec['correct_actions'][:, 1].copy()
        for row, listener in enumerate(self.spec['listener']):
            original = d.env.all_actions(d.env.AGENTS[listener])[int(actions[row, listener])]
            changed = dict(original)
            changed['partner'] = next(a for a in d.env.AGENTS if a not in (d.env.AGENTS[listener], original['partner']))
            actions[row, listener] = d.env.all_actions(d.env.AGENTS[listener]).index(changed)
        result = self.compute(self.fixture(actions=actions))
        self.assertTrue(result['kind_apt'].all() and result['length_apt'].all() and result['destination_apt'].all())
        self.assertFalse(result['partner_apt'].any() or result['unchanged_requirements_apt'].any())

    def test_saved_distribution_and_native_counterfactual_records_are_checked(self):
        for key in ('action_probabilities', 'greedy_reward', 'counterfactual_reward'):
            f = self.fixture(); data = f[3]
            if key == 'action_probabilities':
                data[key] = np.roll(data[key], 1, axis=-1)
            else:
                data[key] = data[key].copy(); data[key][0] = 0 if data[key][0] else 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.compute(f)

    def test_packet_seen_and_message_change_flags_keep_every_row(self):
        f = self.fixture(); data = f[3]
        data['patched_outward_packets'][::2, 0, 0] = 1
        data['messages'][0, 1, self.spec['listener'][0], 0] = 1
        result = self.compute(f)
        np.testing.assert_array_equal(result['packet_in_train_endpoint_greedy_set'], np.arange(144) % 2 == 1)
        self.assertEqual(np.flatnonzero(result['listener_second_message_changed']).tolist(), [0])
        self.assertEqual(np.flatnonzero(result['any_self_generated_message_changed']).tolist(), [0])
        self.assertEqual(len(result['target_apt']), 144)

    def test_outside_packet_gate_keeps_negative_and_original_denominator(self):
        same = dict(target_apt=np.array([1, 0, 0, 1]), natural_current_apt=np.ones(4),
                    packet_in_train_endpoint_greedy_set=np.array([0, 0, 1, 0]))
        opposite = dict(target_apt=np.array([0, 1, 1, 0]), natural_current_apt=np.ones(4),
                        packet_in_train_endpoint_greedy_set=np.array([0, 1, 0, 0]))
        result = m.contrast(same, opposite)
        np.testing.assert_array_equal(result['target_apt'], [-1, 1, 1, -1])
        np.testing.assert_array_equal(result['outside_both_target_apt_gain'], [-1, 0, 0, -1])
        self.assertEqual(result['outside_both_target_apt_gain'].mean(), -.5)
        stacked = m.contrast({k: np.stack([v, v]) for k, v in same.items()},
                             {k: np.stack([v, v]) for k, v in opposite.items()})
        for key in result:
            np.testing.assert_array_equal(stacked[key], np.stack([result[key], result[key]]))
        full_reports = position_reports(self.spec, lambda u: np.full(144, .01 * u))
        full = m.contrast(full_reports[7]['same'], full_reports[7]['opposite'])
        for key in full:
            np.testing.assert_array_equal(full[key], full_reports[7]['contrast'][key])
        bad = dict(opposite); bad['natural_current_apt'] = np.zeros(4)
        with self.assertRaises(ValueError):
            m.contrast(same, bad)

    def test_aggregation_requires_all_balanced_rows_and_both_directions(self):
        values = np.vstack((self.spec['axis'] / 4, self.spec['axis'] / 4 + .1))
        report = m.aggregate_rows(self.spec, {'x': values})
        for a, name in enumerate(m.AXES):
            self.assertAlmostEqual(report['by_axis'][name]['x'], a / 4 + .05)
            for s, l in permutations(range(3), 2):
                self.assertAlmostEqual(report['by_sender_listener'][name][f'{s}_{l}']['x'], a / 4 + .05)
        for bad in (values[:1], values[:, :-1], np.full_like(values, np.nan)):
            with self.assertRaises(ValueError):
                m.aggregate_rows(self.spec, {'x': bad})
        bad = dict(self.spec); bad['base_within_axis_weight'] = np.full(144, 1 / 144)
        with self.assertRaises(ValueError):
            m.aggregate_rows(bad, {'x': values})

    def test_same_samples_selection_and_all_eight_position_baseline(self):
        spec = self.spec
        def effect(u):
            return .01 * u + .02 * (spec['axis'] == u % 3) + .003 * spec['sender']
        reports = position_reports(spec, effect)
        chosen = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 0]])
        result = m.selection_summary(spec, {'selected_positions': chosen}, reports)
        expected = np.zeros((3, 3))
        for q in range(3):
            scores = np.array([effect(int(chosen[s, q]))[r] for r, s in enumerate(spec['sender'])])
            for a in range(3):
                expected[q, a] = scores[spec['axis'] == a].mean()
        np.testing.assert_allclose(result['matrix_rows_discovered_axis_columns_test_axis'], expected, atol=2e-16, rtol=0)
        baseline = np.stack([effect(u) for u in range(8)]).mean()
        self.assertAlmostEqual(result['uniform_position_effect'], baseline)
        self.assertAlmostEqual(result['selectivity'], np.trace(expected) / 3 - (expected.sum() - np.trace(expected)) / 6)
        self.assertAlmostEqual(result['diagonal_minus_uniform_position'], np.trace(expected) / 3 - baseline)
        missing = dict(reports); del missing[7]
        with self.assertRaises(ValueError):
            m.selection_summary(spec, {'selected_positions': chosen}, missing)

    def test_row_and_column_additive_effects_cancel(self):
        spec = self.spec
        effect = lambda u: .013 * u + np.array([-.02, .02, .04])[spec['axis']] + .005 * spec['sender']
        reports = position_reports(spec, effect)
        result = m.selection_summary(spec, {'selected_positions': [[0, 1, 2]] * 3}, reports)
        self.assertAlmostEqual(result['selectivity'], 0, places=15)
        self.assertGreater(result['diagonal_effect'], 0)

    def test_shared_selected_position_has_exact_zero_selectivity(self):
        spec = self.spec
        effect = lambda u: .02 * (u + 1) + .01 * spec['axis'] * (u + 1)
        reports = position_reports(spec, effect)
        # Different senders can choose different positions; q must share per S.
        result = m.selection_summary(spec, {'selected_positions': [[0, 0, 0], [3, 3, 3], [7, 7, 7]]}, reports)
        matrix = np.asarray(result['matrix_rows_discovered_axis_columns_test_axis'])
        np.testing.assert_array_equal(matrix, np.broadcast_to(matrix[0], matrix.shape))
        self.assertEqual(result['selectivity'], 0)
        self.assertGreater(result['diagonal_effect'], 0)

    def test_selection_rejects_changed_controls_unpaired_contrast_or_missing_rows(self):
        selection = {'selected_positions': [[0, 1, 2]] * 3}
        for kind in ('natural', 'contrast', 'shape', 'nonfinite'):
            reports = position_reports(self.spec, lambda u: np.full(144, .01 * u))
            if kind == 'natural':
                reports[7]['same']['natural_current_apt'][0, 0] = 0
            elif kind == 'contrast':
                reports[7]['contrast']['target_apt'][0, 0] += .1
            elif kind == 'shape':
                reports[7]['same']['target_apt'] = reports[7]['same']['target_apt'][:, :-1]
            else:
                reports[7]['opposite']['target_apt'][0, 0] = np.nan
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                m.selection_summary(self.spec, selection, reports)


if __name__ == '__main__':
    unittest.main()
