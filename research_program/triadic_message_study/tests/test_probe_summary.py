"""Tiny fabricated numerical records; no actual run results or policy calls."""
from copy import deepcopy
from itertools import product
import unittest
import numpy as np

from research_program.triadic_message_study import probe_summary as s


def fabricated_content(silent=False):
    ai, aj = (1, 1) if silent else (1, 2)
    definition = dict(pair_id='p', listener='A', listener_index=0,
        row_index_low=0, row_index_high=1, state_low=[0]*10, state_high=[1]*10,
        fullsuccess_actions_low=[1], fullsuccess_actions_high=[2])
    natural = dict(states=np.array([[0]*10, [1]*10]),
        action_indices=np.array([[ai, 0, 0], [aj, 0, 0]]))
    natural['action_probabilities'] = np.eye(17)[natural['action_indices']]
    record = dict(pair_id='p', listener='A', natural_action_low=ai, natural_action_high=aj,
        natural_suitable_low=True, natural_suitable_high=aj == 2,
        natural_both_suitable=aj == 2, natural_action_changed=ai != aj, directions=[])
    for d, own, donor in ((0, 0, 1), (1, 1, 0)):
        action = int(natural['action_indices'][donor, 0])
        record['directions'].append(dict(direction='low_from_high' if d == 0 else 'high_from_low',
            transplanted_action=action, action_changed=ai != aj,
            suitable_for_recipient_world=action == (1 if d == 0 else 2),
            suitable_for_donor_world=action == (2 if d == 0 else 1),
            probability_total_variation=float(ai != aj), donor_probability_saved_exact=True,
            donor_probability_saved_max_abs=0.0, same_batch_donor_probability_exact=True))
    transplant = dict(pair_id=np.array(['p']), probabilities=natural['action_probabilities'][[1, 0], 0])
    return [record], [definition], transplant, natural


class SummaryTests(unittest.TestCase):
    def test_fullsuccess_gains_losses_can_cancel_in_rate(self):
        states = np.array([[0, 0, 0, 0, 1, 2, 3, 1, 2, 3]]*2)
        actions = np.array([[1, 1, 0], [0, 0, 0]])
        changed = actions[::-1].copy()
        natural = dict(states=states, action_indices=actions, action_probabilities=np.eye(17)[actions],
            greedy_reward=np.array([1., 0.]), executed=np.array([[True, True, False], [False]*3]),
            messages=np.zeros((2, 2, 3, 4), dtype=np.int8))
        data = dict(states=states, natural_action_indices=actions)
        row = dict(seed=47101, condition='PI_live', partition='synthetic', worlds=2,
            natural_reward_mean=.5, natural_full_success_rate=.5, deletions={})
        for mode in s.MODES:
            data.update({mode+'_actions':changed, mode+'_probabilities':np.eye(17)[changed],
                mode+'_messages':natural['messages'], mode+'_reward':np.array([0., 1.])})
            row['deletions'][mode] = dict(worlds=2, reward_sum=1., full_successes=1,
                full_success_rate=.5, reward_mean=.5, full_success_rate_change_from_natural=0.,
                worlds_with_any_action_change=2, agent_action_changes=[2, 2, 0],
                fullsuccess_lost=1, fullsuccess_gained=1, executed_transports=2)
        flat = s.summarize_deletions(row, data, natural)
        self.assertEqual(len(flat), 3)
        self.assertEqual(flat[0]['fullsuccess_lost'], flat[0]['fullsuccess_gained'])
        self.assertEqual(flat[0]['world_action_change_rate'], 1.)
        broken = deepcopy(row)
        broken['deletions'][s.MODES[0]]['fullsuccess_lost'] = 0
        with self.assertRaisesRegex(ValueError, 'fullsuccess_lost'):
            s.summarize_deletions(broken, data, natural)

    def test_two_suitable_endpoints_separate_from_identity(self):
        args = fabricated_content()
        records = s.summarize_content(*args, silent=False)
        counts = s.content_counts(records)
        self.assertEqual(counts['naturally_both_suitable'], 1)
        self.assertEqual(counts['natural_action_changes'], 1)
        for direction in counts['directions'].values():
            self.assertEqual(direction['donor_suitable'], 1)
            self.assertEqual(direction['recipient_suitable'], 0)
            self.assertEqual(direction['saved_donor_probability_exact'], 1)

    def test_silent_identity_can_hold_without_both_suitable(self):
        args = fabricated_content(silent=True)
        records = s.summarize_content(*args, silent=True)
        counts = s.content_counts(records)
        self.assertEqual(counts['naturally_both_suitable'], 0)
        self.assertEqual(counts['naturally_exactly_one_suitable'], 1)
        self.assertEqual(counts['natural_action_changes'], 0)
        self.assertEqual(counts['directions']['low_from_high']['saved_donor_probability_exact'], 1)

    def test_wrong_direction_and_missing_pair_rejected(self):
        args = list(fabricated_content())
        args[0][0]['directions'].reverse()
        with self.assertRaisesRegex(ValueError, 'Direction order'):
            s.summarize_content(*args, silent=False)
        args = list(fabricated_content())
        args[0] = []
        with self.assertRaisesRegex(ValueError, 'denominator'):
            s.summarize_content(*args, silent=False)

    def test_wrong_donor_and_fabricated_suitability_rejected(self):
        args = list(fabricated_content())
        args[2]['probabilities'][0] = np.eye(17)[3]
        with self.assertRaisesRegex(ValueError, 'donor probability'):
            s.summarize_content(*args, silent=False)
        args = list(fabricated_content())
        args[0][0]['natural_both_suitable'] = False
        with self.assertRaisesRegex(ValueError, 'suitability'):
            s.summarize_content(*args, silent=False)

    def test_invalid_probability_and_argmax_are_rejected(self):
        with self.assertRaises(ValueError):
            s.validate_probabilities(np.full((1, 17), np.nan), np.array([0]), 'test')
        with self.assertRaises(ValueError):
            s.validate_probabilities(np.eye(17)[[1]], np.array([0]), 'test')

    def test_four_seed_aggregation_keeps_all_cells(self):
        flat, content = [], []
        for seed, condition, partition, mode in product(s.SEEDS, s.CONDITIONS, s.PARTITIONS, s.MODES):
            value = (seed-s.SEEDS[0])/10
            flat.append(dict(seed=seed, condition=condition, partition=partition, mode=mode,
                worlds=s.WORLD_COUNTS[partition], natural_full_success_rate=.5,
                full_success_rate=value, full_success_rate_change_from_natural=value-.5,
                natural_reward_mean=.6, reward_mean=value, world_action_change_rate=value,
                fullsuccess_lost=0, fullsuccess_gained=0))
        for seed, condition, partition in product(s.SEEDS, ('PI_silent', 'PI_live'), s.PARTITIONS):
            content.append(dict(seed=seed, condition=condition, partition=partition,
                all_listeners=dict(natural_both_suitable_rate=.25, natural_action_change_rate=.5)))
        grouped, cg = s.aggregate(flat, content)
        self.assertEqual((len(flat), len(grouped), len(cg)), (192, 48, 8))
        self.assertEqual([r['seed'] for r in grouped[0]['seeds']], list(s.SEEDS))
        self.assertAlmostEqual(grouped[0]['equal_seed_means']['full_success_rate'], .15)
        with self.assertRaisesRegex(ValueError, 'four-seed'):
            s.aggregate(flat[1:], content)


if __name__ == '__main__':
    unittest.main()
