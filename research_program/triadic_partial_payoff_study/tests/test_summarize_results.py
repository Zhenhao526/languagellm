"""Synthetic/native-enumeration fixtures only; no formal output or model reads."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research_program.triadic_partial_payoff_study import summarize_results as s


def small_spec():
    selected = {}
    for needs in s.native.env.support():
        i, j, _, _ = s.native.env.full_success_plans(needs)[0]
        selected.setdefault((i, j), list(needs))
        if len(selected) == 3:
            break
    return dict(needs=[selected[p] for p in sorted(selected)],
                layouts=[[2, 0, 3, 1], [1, 3, 0, 2]], private_sites=[[3, 1, 2]],
                world_count=6, monitor_indices=[0, 2, 4])


def make_saved_record(path, spec, actions=None, alpha=.1, information='FI', live=False):
    ids = np.arange(spec['world_count'], dtype=np.int64)
    states = s.expected_states(spec, ids).astype(np.int16)
    actions = s.native.truth_actions(states) if actions is None else np.asarray(actions, dtype=np.int16)
    point, derived = s.world_summary(states, actions, alpha)
    reward = derived['reward']; full = (reward == 1).astype(float)
    physical = derived['executed'].any(1).astype(float)
    utility = derived['utility']
    rho = np.divide(full, utility, out=np.zeros(len(states)), where=utility > 0)
    data = dict(states=states, state_indices=ids, action_indices=actions,
                action_probabilities=np.eye(17)[actions], messages=np.zeros((len(states), 2, 3, 4), dtype=np.int8),
                greedy_reward=reward, greedy_utility=utility, executed=derived['executed'], satisfied=derived['satisfied'],
                conditional_exact_expected_reward=reward, conditional_exact_full_success_probability=full,
                conditional_exact_execution_probability=physical, conditional_exact_expected_utility=utility,
                conditional_full_posterior_mass=rho)
    np.savez_compressed(path, **data)
    record = dict(path=str(path.resolve()), data_sha256=s.sha(path), worlds=len(states),
                  information=information, live=live, partial_utility=alpha,
                  state_indices_sha256=s.array_sha(ids), reused_natural=False,
                  raw_joint_action_counts=point['all_actual_joint_action_counts'],
                  raw_joint_role_counts=[r for r in point['role_description']['all_27_joint_role_counts'] if r['worlds']])
    for key in ('reward_mean', 'utility_mean', 'role_success_rate', 'full_success_rate', 'physical_execution_rate'):
        record[key] = point[key]
    for key, label in s.MEAN_FIELDS.items():
        if label != 'execution_probability_given_greedy_messages':
            record[label] = float(data[key].mean())
    return record, data


class DescriptionTests(unittest.TestCase):
    def setUp(self):
        self.spec = small_spec()
        self.states = s.expected_states(self.spec, np.arange(6))
        self.truth = s.native.truth_actions(self.states)

    def test_truth_uses_actual_layout_and_each_pair(self):
        expected = []
        for row in self.states:
            state = s.native.env.State(tuple(row[:3]), tuple(row[3:7]), tuple(row[7:10]))
            i, j, site, dest = s.native.env.full_success_plans(state.needs, state.layout)[0]
            expected.append(s.native._plan_actions(i, j, site, dest))
        np.testing.assert_array_equal(self.truth, expected)
        report, derived = s.world_summary(self.states, self.truth, .1)
        self.assertEqual(report['role_success_rate'], 1)
        self.assertEqual(report['full_success_rate'], 1)
        self.assertEqual([v['worlds'] for v in report['truth_pair_strata'].values()], [2, 2, 2])
        self.assertEqual(report['role_description']['observed_reciprocal_pairs'], ['AB', 'AC', 'BC'])
        self.assertIsNone(report['role_description']['all_worlds_same_reciprocal_pair'])
        self.assertEqual(sum(r['worlds'] for r in report['role_description']['all_27_joint_role_counts']), 6)
        self.assertEqual(len(report['role_description']['all_27_joint_role_counts']), 27)

    def test_reciprocal_roles_are_not_physical_execution(self):
        actions = self.truth.copy()
        who = int(np.flatnonzero(actions[0])[0])
        actions[0, who] = 1 + 4 * (((actions[0, who] - 1) // 4 + 1) % 4) + (actions[0, who] - 1) % 4
        report, _ = s.world_summary(self.states, actions, .5)
        self.assertEqual(report['role_success_rate'], 1)
        self.assertEqual(report['full_success_rate'], 5 / 6)
        self.assertEqual(report['physical_execution_rate'], 5 / 6)
        self.assertEqual(report['role_description']['reciprocal_pair_counts']['invalid_role_configuration'], 0)

    def test_partial_payoff_does_not_change_native_reward_or_roles(self):
        actions = self.truth.copy()
        found = False
        for row in s.native.STRUCTURAL_ACTIONS:
            if s.native.settle_arrays(self.states[:1], np.asarray(row)[None])['reward'][0] == .5:
                actions[0] = row; found = True; break
        self.assertTrue(found)
        first, _ = s.world_summary(self.states, actions, .5)
        second, _ = s.world_summary(self.states, actions, .1)
        self.assertEqual(first['reward_mean'], second['reward_mean'])
        self.assertAlmostEqual(first['utility_mean'] - second['utility_mean'], .4 / 6)
        self.assertEqual(first['role_description'], second['role_description'])

    def test_fixed_pair_switch_is_not_task_adaptation(self):
        before = np.repeat(s.PAIR_CODES[0], 6)
        after = np.repeat(s.PAIR_CODES[1], 6)
        truth = s.role_codes(s.native.PARTNER[np.arange(3), self.truth])
        report = s.role_transition(before, after, truth)
        self.assertTrue(report['only_changed_from_one_all_worlds_fixed_pair_to_another'])
        self.assertEqual(report['joint_role_changed_worlds'], 6)
        self.assertEqual(report['role_success_rate_difference'], 0)
        self.assertEqual(report['newly_role_correct_worlds'], 2)
        self.assertEqual(report['newly_role_incorrect_worlds'], 2)
        self.assertEqual(report['actors']['A']['changed_partner_while_active_in_both_worlds'], 6)
        self.assertEqual(report['actors']['B']['changed_partner_while_active_in_both_worlds'], 0)
        report = s.role_transition(before, truth, truth)
        self.assertFalse(report['only_changed_from_one_all_worlds_fixed_pair_to_another'])
        self.assertEqual(report['newly_role_correct_worlds'], 4)

    def test_invalid_roles_and_partial_pair_population_not_hidden(self):
        codes = np.asarray([s.PAIR_CODES[0], s.PAIR_CODES[0], 0], dtype=np.uint8)
        report = s.role_description(codes)
        self.assertIsNone(report['all_worlds_same_reciprocal_pair'])
        self.assertEqual(report['only_one_reciprocal_pair_when_roles_form_pair'], 'AB')
        self.assertEqual(report['reciprocal_pair_counts']['invalid_role_configuration'], 1)
        with self.assertRaisesRegex(ValueError, 'self partner'):
            s.role_codes(np.array([[0, -1, -1]]))
        with self.assertRaisesRegex(ValueError, 'align'):
            s.role_transition(codes, codes[:2], codes)

    def test_actual_npz_summarized_and_bad_truth_layout_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fixture.npz'
            record, data = make_saved_record(path, self.spec)
            bound = {}
            report, codes, truth = s.summarize_record(record, self.spec, np.arange(6), .1, path, bound)
            self.assertEqual(report['full_success_rate'], 1)
            self.assertEqual(report['saved_expectation_means']['full_posterior_mass_given_greedy_messages'], 1)
            self.assertEqual(len(bound), 1)
            changed = deepcopy(record); changed['raw_joint_role_counts'][0]['worlds'] += 1
            with self.assertRaisesRegex(ValueError, 'role frequencies'):
                s.summarize_record(changed, self.spec, np.arange(6), .1, path, {})
            bad = data.copy(); bad['states'] = bad['states'].copy(); bad['states'][0, [3, 4]] = bad['states'][0, [4, 3]]
            np.savez_compressed(path, **bad); changed = deepcopy(record); changed['data_sha256'] = s.sha(path)
            with self.assertRaisesRegex(ValueError, 'Packed worlds'):
                s.summarize_record(changed, self.spec, np.arange(6), .1, path, {})

    def test_per_world_posterior_mean_and_array_identity(self):
        # Arithmetic fixture includes heterogeneous probabilities; mean(F/U)
        # must not be replaced by mean(F)/mean(U).
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fixture.npz'; record, data = make_saved_record(path, self.spec)
            full = np.array([.1, .4, .2, .3, .1, .2]); partial = np.array([.5, .1, .3, .1, .6, .2])
            data['conditional_exact_full_success_probability'] = full
            data['conditional_exact_expected_reward'] = full + .5 * partial
            data['conditional_exact_expected_utility'] = full + .1 * partial
            data['conditional_exact_execution_probability'] = full + partial
            data['conditional_full_posterior_mass'] = full / (full + .1 * partial)
            for key, label in s.MEAN_FIELDS.items():
                if label in record: record[label] = float(data[key].mean())
            np.savez_compressed(path, **data); record['data_sha256'] = s.sha(path)
            report, _, _ = s.summarize_record(record, self.spec, np.arange(6), .1, path, {})
            self.assertEqual(report['saved_expectation_means']['full_posterior_mass_given_greedy_messages'], float(data['conditional_full_posterior_mass'].mean()))
            self.assertNotAlmostEqual(float(data['conditional_full_posterior_mass'].mean()), float(full.mean() / data['conditional_exact_expected_utility'].mean()))
            # Confirm exact original serialization including metadata newline.
            from research_program.triadic_learning_baseline.runner import array_sha as old_sha
            self.assertEqual(s.array_sha(data['state_indices']), old_sha(data['state_indices']))

    def test_refuses_unfinished_before_reading_npz_or_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'execution').mkdir()
            (root / 'execution/status.json').write_text(json.dumps({'status': 'running'}))
            out = root / 'summary'
            with patch.object(s.np, 'load', side_effect=AssertionError('NPZ must not be read')):
                with self.assertRaisesRegex(ValueError, 'not completed'):
                    s.summarize(root, out)
            self.assertFalse(out.exists())


class SemanticIntegrationTests(unittest.TestCase):
    def test_complete_native_fixture_has_all_axes_and_sender_listener_strata(self):
        # Full *static task support* at one actual layout, generated without any
        # saved policy result. Oracle action arrays are test labels, not agents.
        spec = dict(needs=[list(n) for n in s.native.env.support()], layouts=[[2, 0, 3, 1]], private_sites=[[1, 2, 3]])
        spec['world_count'] = len(spec['needs'])
        pairs = s.native.dataset.content_pairs(spec)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fixture.npz'; record, _ = make_saved_record(path, spec)
            report, _, _ = s.summarize_record(record, spec, np.arange(spec['world_count']), .1, path, {}, semantic=True, pairs=pairs)
            semantic = report['semantic_metrics']
            self.assertEqual(semantic['coverage'], 'full_partition')
            for family in ('content', 'role'):
                self.assertEqual(semantic[family]['macro']['both_endpoints_apt'], 1)
                self.assertEqual(set(semantic[family]['by_axis']), {'kind', 'length', 'destination'})
                for axis in semantic[family]['by_axis'].values():
                    self.assertTrue(axis['complete_six_strata'])
                    self.assertEqual(len(axis['strata']), 6)


if __name__ == '__main__':
    unittest.main()
