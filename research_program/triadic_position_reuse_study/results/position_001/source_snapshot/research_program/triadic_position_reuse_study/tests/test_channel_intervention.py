"""Independent routing predicates and handwritten fake forwards, no learned actor."""
from itertools import product, permutations
from unittest import TestCase, mock

import numpy as np

from research_program.triadic_position_reuse_study import channel_intervention as c


def fake_forward(net, x):
    value = np.rint(x @ (1 + np.arange(x.shape[1]) % 7)).astype(int)
    if net % 3 == 2:
        z = np.full((len(x), 17), -4.)
        z[np.arange(len(x)), (value + net) % 17] = 4.
    else:
        z = np.full((len(x), 4, 8), -4.)
        for t in range(4):
            z[np.arange(len(x)), t, (value + net + t) % 8] = 4.
        z = z.reshape(len(x), 32)
    return z, None


def natural(x, live=True):
    result = c.core.rollout(tuple(range(9)), x, live)
    probability, _ = c.core.base.policy_distribution(result['action_logits'])
    return dict(messages=result['messages'], action_inputs=result['action_inputs'],
                action_probabilities=probability, action_indices=probability.argmax(-1))


class PositionChannelTests(TestCase):
    def test_all_sender_position_symbol_routes_keep_self_and_visibility(self):
        cases = list(product(range(3), range(4), range(8)))
        n = len(cases)
        senders, positions, symbols = map(np.asarray, zip(*cases))
        tokens = np.arange(n*3*4).reshape(n, 3, 4) % 8
        before = tokens.copy()
        got = c.replace_outward(tokens, senders, positions, symbols, True)
        expected = np.zeros((n, 3, 99))
        for row, (sender, position, symbol) in enumerate(cases):
            for viewer in range(3):
                for source in range(3):
                    for slot in range(4):
                        value = symbol if (viewer != sender and source == sender and slot == position) else tokens[row, source, slot]
                        expected[row, viewer, 32*source+8*slot+value] = 1.
                expected[row, viewer, 96:] = 1.
        np.testing.assert_array_equal(got, expected)
        np.testing.assert_array_equal(tokens, before)

    @mock.patch.object(c.core.base, 'actor_forward', side_effect=fake_forward)
    def test_sham_for_both_windows_replays_natural_and_counts_samples(self, forward):
        x = np.random.default_rng(271).integers(0, 2, (6, 3, 54)).astype(float)
        original = natural(x)
        senders = np.tile(np.arange(3), 2)
        windows = np.repeat([0, 1], 3)
        donor = original['messages'][np.arange(6), :, senders]
        forward.reset_mock()
        got = c.intervene(tuple(range(9)), x, original['messages'], senders, donor, windows, 2)
        for key in ('messages', 'action_inputs', 'action_indices', 'action_probabilities'):
            np.testing.assert_array_equal(got[key], original[key])
        np.testing.assert_array_equal(got['patched_outward_packets'], donor)
        self.assertEqual(forward.call_count, 6)
        self.assertEqual([len(call.args[1]) for call in forward.call_args_list], [3, 3, 3, 6, 6, 6])
        self.assertEqual(got['second_window_forward_samples'], 9)
        self.assertEqual(got['action_forward_samples'], 18)
        self.assertEqual(got['neural_forward_samples'], 27)
        for key, value in got['routing_sha256'].items():
            self.assertEqual(value, c.core.array_sha(got[key]))

    def test_w1_feedback_changes_non_sender_second_messages_before_action(self):
        # Action heads ignore W1 and read only their own self-generated W2 token.
        # Any changed action therefore requires correctly recomputing that W2.
        def mediated(net, x):
            agent = net // 3
            if net % 3 == 1:
                symbol = np.argmax(x[:, 54:62], axis=-1)  # sender A's first token
                z = np.full((len(x), 4, 8), -9.)
                z[:, :, 0] = 9.
                z[:, 0, :] = -9.
                z[np.arange(len(x)), 0, symbol] = 9.
                return z.reshape(len(x), 32), None
            own_second = 54+99+32*agent
            symbol = np.argmax(x[:, own_second:own_second+8], axis=-1)
            z = np.full((len(x), 17), -9.)
            z[np.arange(len(x)), symbol] = 9.
            return z, None
        x = np.zeros((1, 3, 54))
        messages = np.zeros((1, 2, 3, 4), dtype=np.int8)
        donor = np.full((1, 2, 4), 7, dtype=np.int8)
        donor[0, 0, 0] = 1
        with mock.patch.object(c.core.base, 'actor_forward', side_effect=mediated) as forward:
            got = c.intervene(tuple(range(9)), x, messages, 0, donor, 0, 0)
        self.assertEqual(forward.call_count, 6)
        np.testing.assert_array_equal(got['messages'][:, 0], messages[:, 0])
        np.testing.assert_array_equal(got['messages'][0, :, 0], messages[0, :, 0])
        np.testing.assert_array_equal(got['messages'][0, 1, 1:, 0], [1, 1])
        np.testing.assert_array_equal(got['action_indices'][0], [0, 1, 1])
        expected_packet = np.zeros((1, 2, 4), dtype=np.int8)
        expected_packet[0, 0, 0] = 1
        np.testing.assert_array_equal(got['patched_outward_packets'], expected_packet)
        # Nonselected donor sevens never enter any outgoing sender-A slot.
        for viewer in (1, 2):
            np.testing.assert_array_equal(got['second_routes'][0, viewer, :32], np.eye(8)[[0]*4].reshape(32))

    @mock.patch.object(c.core.base, 'actor_forward', side_effect=fake_forward)
    def test_w2_splice_keeps_all_history_and_runs_only_action_heads(self, forward):
        x = np.zeros((3, 3, 54))
        original = natural(x)
        m = original['messages']; before = m.copy()
        senders = np.arange(3); positions = np.array([0, 2, 3])
        donor = (m[np.arange(3), :, senders]+1) % 8
        before_donor = donor.copy()
        forward.reset_mock()
        got = c.intervene(tuple(range(9)), x, m, senders, donor, 1, positions)
        self.assertEqual([call.args[0] for call in forward.call_args_list], [2, 5, 8])
        self.assertEqual(got['neural_forward_samples'], 9)
        self.assertEqual(got['second_window_forward_samples'], 0)
        np.testing.assert_array_equal(got['messages'], before)
        np.testing.assert_array_equal(got['first_routes'], original['action_inputs'][:, :, 54:153])
        for row, sender in enumerate(senders):
            np.testing.assert_array_equal(got['action_inputs'][row, sender], original['action_inputs'][row, sender])
            for viewer in range(3):
                if viewer == sender:
                    continue
                different = np.flatnonzero(got['second_routes'][row, viewer] != original['action_inputs'][row, viewer, 153:])
                lo = 32*sender+8*positions[row]
                self.assertEqual(len(different), 2)
                self.assertTrue(np.all((different >= lo) & (different < lo+8)))
        np.testing.assert_array_equal(m, before)
        np.testing.assert_array_equal(donor, before_donor)

    @mock.patch.object(c.core.base, 'actor_forward', side_effect=fake_forward)
    def test_mixed_rows_match_separate_calls_with_heterogeneous_controls(self, forward):
        x = np.random.default_rng(919).integers(0, 2, (4, 3, 54)).astype(float)
        original = natural(x)
        m = original['messages']; before = m.copy()
        s = np.array([0, 2, 1, 0]); w = np.array([0, 1, 0, 1]); p = np.arange(4)
        donor = (m[np.arange(4), :, s]+3) % 8
        forward.reset_mock()
        batched = c.intervene(tuple(range(9)), x, m, s, donor, w, p)
        self.assertEqual([len(call.args[1]) for call in forward.call_args_list], [2, 2, 2, 4, 4, 4])
        self.assertEqual(batched['neural_forward_samples'], 18)
        single = [c.intervene(tuple(range(9)), x[i:i+1], m[i:i+1], int(s[i]), donor[i:i+1], int(w[i]), int(p[i])) for i in range(4)]
        for key in ('messages', 'patched_outward_packets', 'first_routes', 'second_routes',
                    'action_inputs', 'action_indices', 'action_probabilities'):
            np.testing.assert_array_equal(batched[key], np.concatenate([r[key] for r in single]))
        np.testing.assert_array_equal(m, before)

    def test_silent_alias_checks_routes_with_no_networks_or_forwards(self):
        x = np.zeros((4, 3, 54)); m = np.arange(4*2*3*4).reshape(4, 2, 3, 4) % 8
        sender = np.array([0, 1, 2, 1]); donor = np.full((4, 2, 4), 7, dtype=int)
        with mock.patch.object(c.core.base, 'actor_forward', side_effect=AssertionError('No NN allowed')):
            got = c.intervene(None, x, m, sender, donor, [0, 1, 0, 1], [3, 2, 1, 0], False)
        np.testing.assert_array_equal(got['messages'], m)
        np.testing.assert_array_equal(got['action_inputs'], c.natural_action_inputs(x, m, False))
        self.assertEqual(got['neural_forward_samples'], 0)
        self.assertEqual(got['neural_forward_calls'], 0)
        self.assertTrue(got['reused_natural'])
        self.assertFalse(got['outward_patch_visible'])
        self.assertNotIn('action_probabilities', got)
        self.assertNotIn('action_indices', got)

    def test_softmax_rounding_ties_for_w1_and_both_action_branches(self):
        def tiny(net, x):
            z = np.zeros((len(x), 32 if net % 3 == 1 else 17))
            z[:, 1::8 if net % 3 == 1 else 100] = 1e-18
            return z, None
        with mock.patch.object(c.core.base, 'actor_forward', side_effect=tiny):
            got = c.intervene(tuple(range(9)), np.zeros((2, 3, 54)),
                np.zeros((2, 2, 3, 4), dtype=np.int8), [0, 1],
                np.full((2, 2, 4), 7, dtype=np.int8), [0, 1], [0, 3])
        np.testing.assert_array_equal(got['messages'][0, 1], 0)
        np.testing.assert_array_equal(got['action_indices'], 0)

    @mock.patch.object(c.core.base, 'actor_forward', side_effect=fake_forward)
    def test_unseen_mixed_packet_is_legal_and_nonselected_donor_tokens_irrelevant(self, forward):
        x = np.zeros((1, 3, 54)); original = natural(x); m = original['messages']
        donor = m[:, :, 1].copy()
        donor[0, 1, 2] = (donor[0, 1, 2]+1) % 8
        got = c.intervene(tuple(range(9)), x, m, 1, donor, 1, 2)
        self.assertTrue(np.any(got['patched_outward_packets'] != m[:, :, 1]))
        other = (donor+3) % 8; other[0, 1, 2] = donor[0, 1, 2]
        same = c.intervene(tuple(range(9)), x, m, 1, other, 1, 2)
        for key in ('patched_outward_packets', 'messages', 'action_inputs', 'action_probabilities'):
            np.testing.assert_array_equal(got[key], same[key])

    def test_invalid_symbols_shapes_and_selectors_rejected_before_forward(self):
        args = dict(networks=tuple(range(9)), observations=np.zeros((1, 3, 54)),
                    natural_messages=np.zeros((1, 2, 3, 4), dtype=int), senders=0,
                    donor_packets=np.zeros((1, 2, 4), dtype=int), windows=0, positions=0)
        bad = [('donor_packets', np.full((1, 2, 4), 8)), ('donor_packets', np.zeros((1, 2, 4))),
               ('senders', True), ('senders', 3), ('windows', 2), ('windows', [0, 1]),
               ('positions', -1), ('positions', 4), ('positions', .0), ('live', 1),
               ('observations', np.full((1, 3, 54), np.nan))]
        with mock.patch.object(c.core.base, 'actor_forward', side_effect=AssertionError('Invalid inputs reached NN')):
            for key, value in bad:
                with self.subTest(key=key, value=str(value)):
                    with self.assertRaises(ValueError):
                        c.intervene(**dict(args, **{key: value}))

    def test_official_24_need_encoder_and_native_settlement_wrappers(self):
        needs = next(n for n in c.env.support() if max(n) >= 12)
        rows, acts = [], []
        for layout in permutations(range(4)):
            state = c.env.State(needs, layout, (3, 1, 2))
            rows.append(state.needs+state.layout+state.private_sites)
            plan = c.env.sufficient_information_witness(state)
            acts.append([c.env.all_actions(a).index(plan[a]) for a in c.env.AGENTS])
        rows, acts = np.asarray(rows), np.asarray(acts)
        pl, ll = c.observations(rows, 'PL'), c.observations(rows, 'LL')
        self.assertEqual(pl.shape, (24, 3, 54))
        self.assertTrue(np.any(pl != ll))
        np.testing.assert_array_equal(c.settle(rows, acts)['greedy_reward'], 1)
        with self.assertRaises(ValueError):
            c.observations(rows, 'FI')
