"""Synthetic tests only: no source dataset, learned policy or model loaded."""
import copy
import json
import unittest
import numpy as np
from research_program.triadic_semantic_probe_study import metrics as m


def fixture(include_role=False):
    # Unequal axis row counts 1,1,3: equal-row averaging would be incorrect.
    n = 7 if include_role else 5
    labels = dict(success_action_masks=np.tile(np.array([2, 4], dtype=np.uint32), (n, 1)),
        listener=np.ones(n, dtype=np.int8), sender=np.zeros(n, dtype=np.int8),
        axis=np.array([0, 1, 2, 2, 2]+([0, 1] if include_role else []), dtype=np.int8),
        listener_partners=np.zeros((n, 2), dtype=np.int8),
        classification=np.array([1]*5+([2]*2 if include_role else []), dtype=np.int8),
        content_within_axis_weight=np.array([1, 1, 1/3, 1/3, 1/3]+([0, 0] if include_role else [])),
        role_within_axis_weight=np.array([0]*5+([1, 1] if include_role else []), dtype=float))
    # Both endpoints apt for rows 0 and 2 only.
    actions = np.zeros((n, 2, 3), dtype=np.int16)
    actions[[0, 2], :, 1] = [1, 2]
    if include_role:
        labels['listener_partners'][5:] = [-1, 2]
        labels['success_action_masks'][5:] = [1, 4]
        actions[5:, :, 1] = [0, 2]
    return labels, actions, np.eye(17)[actions]


def natural(labels, actions, probs=None):
    return m.natural_metrics(**labels, action_indices=actions,
                             action_probs=np.eye(17)[actions] if probs is None else probs)


def intervention(labels, actions, reference, probs=None, window='both', rewards=None):
    return m.intervention_metrics(**labels, action_indices=actions,
        action_probs=np.eye(17)[actions] if probs is None else probs,
        natural_action_indices=reference, natural_action_probs=np.eye(17)[reference],
        native_rewards=np.zeros(actions.shape[:2]) if rewards is None else rewards, window=window)


class MetricsTests(unittest.TestCase):
    def test_three_axis_macro_not_pooled_rows(self):
        labels, a, p = fixture()
        result = natural(labels, a, p)
        self.assertAlmostEqual(result['content']['macro']['both_endpoints_apt'], 4/9)
        self.assertEqual(result['all_rows_unweighted']['metrics']['both_endpoints_apt']['mean'], .4)
        self.assertEqual(result['content']['raw_unweighted']['n_case_worlds'], 5)
        self.assertEqual(result['content']['raw_unweighted']['n_endpoint_directions'], 10)
        self.assertEqual(result['row_values']['both_endpoints_apt'].tolist(), [True, False, True, False, False])

    def test_role_not_part_of_primary(self):
        labels, a, _ = fixture(include_role=True)
        result = natural(labels, a)
        self.assertAlmostEqual(result['content']['macro']['both_endpoints_apt'], 4/9)
        self.assertEqual(result['role']['macro']['both_endpoints_apt'], 1)
        self.assertEqual(result['role']['macro_axes'], list(m.AXES[:2]))
        self.assertEqual(result['role']['by_axis'][m.AXES[0]]['n_case_worlds'], 1)
        self.assertEqual(result['all_rows_unweighted']['n_case_worlds'], 7)

    def test_saved_mass_and_endpoint_means(self):
        labels, a, _ = fixture()
        p = np.eye(17)[a]*.7
        # Add .3 at an action outside every accepted mask, preserving argmax.
        p[..., 16] += .3
        result = natural(labels, a, p)
        np.testing.assert_allclose(result['row_values']['endpoint_probability_mass'][[0, 2]], .7)
        self.assertAlmostEqual(result['content']['macro']['mean_endpoint_probability_mass'], .7*4/9)
        self.assertEqual(result['content']['macro']['listener_action_changed'], 4/9)

    def test_within_axis_weights_used(self):
        labels, a, _ = fixture()
        labels['content_within_axis_weight'][2:] = [.5, .25, .25]
        result = natural(labels, a)
        self.assertEqual(result['content']['by_axis'][m.AXES[2]]['metrics']['both_endpoints_apt']['mean'], .5)
        self.assertEqual(result['content']['macro']['both_endpoints_apt'], .5)

    def test_bidirectional_counterfactual_recipient_masks(self):
        labels, a, _ = fixture()
        reference = a.copy(); reference[:, :, 1] = [1, 2]
        changed = reference.copy(); changed[:, :, 1] = [2, 1]
        result = intervention(labels, changed, reference)
        np.testing.assert_array_equal(result['row_values']['counterfactual_apt'], True)
        np.testing.assert_array_equal(result['row_values']['current_apt'], False)
        self.assertEqual(result['content']['macro']['direction_mean_counterfactual_apt'], 1)
        self.assertEqual(result['content']['macro']['direction_mean_native_reward'], 0)
        self.assertEqual(result['window_status'], 'predetermined_secondary')

    def test_one_direction_change_is_one_half_not_second_replicate(self):
        labels, a, _ = fixture()
        reference = a.copy(); reference[:, :, 1] = [1, 2]
        changed = reference.copy(); changed[:, 0, 1] = 2
        result = intervention(labels, changed, reference, window='w1')
        self.assertEqual(result['content']['macro']['counterfactual_apt'], [1, 0])
        self.assertEqual(result['content']['macro']['direction_mean_counterfactual_apt'], .5)
        self.assertEqual(result['content']['raw_unweighted']['n_case_worlds'], 5)

    def test_remote_signed_contrast_and_loss(self):
        labels, a, _ = fixture()
        reference = a.copy(); reference[:, :, 1] = [1, 2]
        same_actions = reference.copy(); same_actions[0, :, 1] = 0
        opposite_actions = reference.copy(); opposite_actions[:, :, 1] = [2, 1]
        same = intervention(labels, same_actions, reference, rewards=np.ones((5, 2)))
        opposite = intervention(labels, opposite_actions, reference)
        result = m.remote_contrast(same, opposite)
        self.assertEqual(result['content']['macro']['direction_mean_target_apt_opposite_minus_same'], 1)
        self.assertAlmostEqual(result['content']['macro']['direction_mean_current_apt_same_minus_natural'], -1/3)
        self.assertAlmostEqual(result['content']['macro']['direction_mean_current_apt_loss_natural_minus_same'], 1/3)
        self.assertEqual(result['content']['macro']['direction_mean_native_reward_opposite_minus_same'], -1)

    def test_remote_pairing_rejects_changed_window_or_reference(self):
        labels, a, _ = fixture()
        same = intervention(labels, a, a)
        different_window = intervention(labels, a, a, window='w2')
        with self.assertRaises(ValueError): m.remote_contrast(same, different_window)
        b = a.copy(); b[0, 0, 1] = 0
        different_reference = intervention(labels, a, b)
        with self.assertRaises(ValueError): m.remote_contrast(same, different_reference)

    def test_all_labels_and_failures_retained(self):
        labels, a, _ = fixture()
        result = natural(labels, a)
        strata = result['content']['strata']
        self.assertEqual(set(strata['sender_listener_pair']), {'AB', 'AC', 'BC'})
        self.assertEqual(len(strata['ordered_sender_listener']), 6)
        self.assertEqual(strata['sender_listener_pair']['AC']['raw_unweighted']['n_case_worlds'], 0)
        self.assertEqual(strata['sender_listener_pair']['AB']['raw_unweighted']['n_case_worlds'], 5)

    def test_invalid_weights_masks_policy_reward_rejected(self):
        labels, a, p = fixture()
        invalid = copy.deepcopy(labels); invalid['content_within_axis_weight'][0] = .5
        with self.assertRaises(ValueError): natural(invalid, a)
        invalid = copy.deepcopy(labels); invalid['success_action_masks'][0, 1] = 2
        with self.assertRaises(ValueError): natural(invalid, a)
        badp = p.copy(); badp[0, 0, 0, 0] = .5
        with self.assertRaises(ValueError): natural(labels, a, badp)
        badp = p.copy(); badp[0, 0, 0] = np.eye(17)[1]
        with self.assertRaises(ValueError): natural(labels, a, badp)
        with self.assertRaises(ValueError): intervention(labels, a, a, rewards=np.full((5, 2), .25))
        full, fa, _ = fixture(include_role=True)
        with self.assertRaises(ValueError): intervention(full, fa, fa)

    def test_inputs_unchanged(self):
        labels, a, p = fixture(); old = copy.deepcopy((labels, a, p))
        natural(labels, a, p)
        intervention(labels, a, a)
        for k in labels: np.testing.assert_array_equal(labels[k], old[0][k])
        np.testing.assert_array_equal(a, old[1]); np.testing.assert_array_equal(p, old[2])

    def test_native_team_metrics_separate_and_json_summary(self):
        labels, a, p = fixture()
        rewards = np.tile([.5, 1.], (5, 1))
        result = m.natural_metrics(**labels, action_indices=a, action_probs=p, native_rewards=rewards)
        self.assertEqual(result['content']['macro']['mean_native_reward'], .75)
        self.assertEqual(result['content']['macro']['mean_endpoint_team_full_success'], .5)
        self.assertEqual(result['content']['macro']['both_team_full_success'], 0)
        json.dumps(m.summary_only(result), allow_nan=False)
        json.dumps(m.summary_only(intervention(labels, a, a)), allow_nan=False)


if __name__ == '__main__': unittest.main()
