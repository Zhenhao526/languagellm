"""Handwritten fake forwards and independent routes; no saved model or training."""
from unittest import TestCase, mock
import numpy as np

from research_program.triadic_formation_trajectory_study import channel as c


NETWORKS = tuple(range(9))


def fake_forward(net, x):
    value = np.rint(x @ (1+np.arange(x.shape[1]) % 7)).astype(int)
    if net % 3 == 2:
        z = np.full((len(x), 17), -5.)
        z[np.arange(len(x)), (value+net) % 17] = 5.
    else:
        z = np.full((len(x), 4, 8), -5.)
        for p in range(4):
            z[np.arange(len(x)), p, (value+net+p) % 8] = 5.
        z = z.reshape(len(x), 32)
    return z, None


def independent_route(tokens, live):
    out = np.zeros((len(tokens), 3, 99))
    for row in range(len(tokens)):
        for viewer in range(3):
            for source in range(3):
                if live or source == viewer:
                    for p in range(4):
                        out[row, viewer, source*32+p*8+tokens[row, source, p]] = 1
                    out[row, viewer, 96+source] = 1
    return out


class FormationChannelTests(TestCase):
    @mock.patch.object(c.core.base, 'actor_forward', side_effect=fake_forward)
    def test_message_only_has_six_heads_and_complete_window_barrier(self, forward):
        x = np.random.default_rng(813).integers(0, 2, (5, 3, 54)).astype(float)
        before = x.copy()
        for live in (True, False):
            with self.subTest(live=live):
                forward.reset_mock()
                got = c.generate_messages(NETWORKS, x, live, include_routes=True)
                self.assertEqual([call.args[0] for call in forward.call_args_list], [0, 3, 6, 1, 4, 7])
                self.assertEqual([call.args[1].shape for call in forward.call_args_list], [(5, 54)]*3+[(5, 153)]*3)
                first_route = independent_route(got['messages'][:, 0], live)
                np.testing.assert_array_equal(got['first_routes'], first_route)
                np.testing.assert_array_equal(got['second_routes'], independent_route(got['messages'][:, 1], live))
                for agent in range(3):
                    expected_input = np.concatenate((x[:, agent], first_route[:, agent]), -1)
                    np.testing.assert_array_equal(forward.call_args_list[3+agent].args[1], expected_input)
                self.assertEqual(got['neural_forward_samples'], 30)
                self.assertEqual(got['action_forward_samples'], 0)
                self.assertNotIn('action_indices', got)
                self.assertNotIn('action_probabilities', got)
                np.testing.assert_array_equal(x, before)

    @mock.patch.object(c.core.base, 'actor_forward', side_effect=fake_forward)
    def test_natural_rollout_exactly_matches_old_greedy_rollout(self, forward):
        x = np.random.default_rng(7).integers(0, 2, (4, 3, 54)).astype(float)
        for live in (True, False):
            with self.subTest(live=live):
                old = c.core.rollout(NETWORKS, x, live)
                probability, _ = c.core.base.policy_distribution(old['action_logits'])
                forward.reset_mock()
                got = c.natural_rollout(NETWORKS, x, live)
                np.testing.assert_array_equal(got['messages'], old['messages'])
                np.testing.assert_array_equal(got['action_inputs'], old['action_inputs'])
                np.testing.assert_array_equal(got['action_probabilities'], probability)
                np.testing.assert_array_equal(got['action_indices'], probability.argmax(-1))
                self.assertEqual(got['neural_forward_samples'], 36)
                self.assertEqual(got['neural_forward_calls'], forward.call_count)
                self.assertEqual(forward.call_count, 9)
                for name, digest in got['routing_sha256'].items():
                    self.assertEqual(digest, c.core.array_sha(got[name]))
                forward.reset_mock()
                messages = c.generate_messages(NETWORKS, x, live)
                np.testing.assert_array_equal(messages['messages'], got['messages'])
                self.assertNotIn('first_routes', messages)
                self.assertEqual(forward.call_count, 6)

    def test_greedy_uses_softmax_probabilities_before_argmax(self):
        def rounded_tie(net, x):
            z = np.zeros((len(x), 17 if net % 3 == 2 else 32))
            if net % 3 == 2:
                z[:, 1] = 1e-18
            else:
                z[:, 1::8] = 1e-18
            return z, None
        with mock.patch.object(c.core.base, 'actor_forward', side_effect=rounded_tie):
            got = c.natural_rollout(NETWORKS, np.zeros((2, 3, 54)), True)
        np.testing.assert_array_equal(got['messages'], 0)
        np.testing.assert_array_equal(got['action_indices'], 0)

    def test_cross_time_packet_keeps_receiver_sender_self_but_mediates_others(self):
        # Receiver W1=0. Receiver W2 first token copies the received A W1 token;
        # receiver actions read B's W2. Thus even A action may change while its
        # own W1/W2 remain receiver-natural. No donor network exists in this test.
        def mediated(net, x):
            if net % 3 == 0:
                symbols = np.zeros(len(x), dtype=int)
            elif net % 3 == 1:
                symbols = x[:, 54:62].argmax(-1)
            else:
                symbols = x[:, 153+32:153+40].argmax(-1)
                z = np.full((len(x), 17), -9.)
                z[np.arange(len(x)), symbols] = 9.
                return z, None
            z = np.full((len(x), 4, 8), -9.)
            z[:, :, 0] = 9.
            z[:, 0] = -9.
            z[np.arange(len(x)), 0, symbols] = 9.
            return z.reshape(len(x), 32), None
        x = np.zeros((1, 3, 54))
        with mock.patch.object(c.core.base, 'actor_forward', side_effect=mediated) as forward:
            natural = c.natural_rollout(NETWORKS, x, True)
            donor = np.zeros((1, 2, 4), dtype=int)
            donor[0, 0, 0], donor[0, 1, 0] = 1, 7
            before = donor.copy()
            forward.reset_mock()
            got = c.cross_time_whole(NETWORKS, x, natural['messages'], 0, donor)
        self.assertEqual([call.args[0] for call in forward.call_args_list], [1, 4, 7, 2, 5, 8])
        np.testing.assert_array_equal(got['messages'][0, :, 0], natural['messages'][0, :, 0])
        np.testing.assert_array_equal(got['messages'][0, 1, 1:, 0], [1, 1])
        np.testing.assert_array_equal(got['action_indices'], [[1, 1, 1]])
        np.testing.assert_array_equal(natural['action_indices'], 0)
        for viewer in (1, 2):
            np.testing.assert_array_equal(got['first_routes'][0, viewer, :32], np.eye(8)[donor[0, 0]].reshape(32))
            np.testing.assert_array_equal(got['second_routes'][0, viewer, :32], np.eye(8)[donor[0, 1]].reshape(32))
        np.testing.assert_array_equal(got['second_routes'][0, 0, :32], np.eye(8)[[0]*4].reshape(32))
        np.testing.assert_array_equal(donor, before)
        self.assertEqual(got['neural_forward_samples'], 6)

    @mock.patch.object(c.core.base, 'actor_forward', side_effect=fake_forward)
    def test_whole_wrapper_matches_unchanged_whole_and_sham(self, forward):
        x = np.random.default_rng(82).integers(0, 2, (3, 3, 54)).astype(float)
        native = c.natural_rollout(NETWORKS, x, True)
        m = native['messages']; before = m.copy(); s = np.arange(3)
        for sham in (True, False):
            with self.subTest(sham=sham):
                donor = m[np.arange(3), :, s].copy()
                if not sham:
                    donor = (donor+2) % 8
                old = c.whole.intervene(NETWORKS, x, m, s, donor, True)
                got = c.whole_intervention(NETWORKS, x, m, s, donor)
                for name in ('messages', 'action_inputs', 'first_routes', 'second_routes',
                             'action_probabilities', 'action_indices'):
                    np.testing.assert_array_equal(got[name], old[name])
                if sham:
                    for name in ('messages', 'action_inputs', 'action_probabilities', 'action_indices'):
                        np.testing.assert_array_equal(got[name], native[name])
                self.assertEqual(got['neural_forward_samples'], 18)
                np.testing.assert_array_equal(m, before)

    @mock.patch.object(c.core.base, 'actor_forward', side_effect=fake_forward)
    def test_position_wrapper_keeps_original_early_late_costs(self, forward):
        x = np.zeros((3, 3, 54))
        m = c.natural_rollout(NETWORKS, x, True)['messages']
        s = np.arange(3); donor = (m[np.arange(3), :, s]+1) % 8
        forward.reset_mock()
        got = c.position_intervention(NETWORKS, x, m, s, donor, [0, 1, 1], [1, 3, 0])
        self.assertEqual(got['neural_forward_samples'], 12)
        self.assertEqual([len(call.args[1]) for call in forward.call_args_list], [1, 1, 1, 3, 3, 3])
        np.testing.assert_array_equal(got['messages'][1:], m[1:])
        for row in range(3):
            np.testing.assert_array_equal(got['messages'][row, :, s[row]], m[row, :, s[row]])

    def test_silent_whole_and_position_return_exact_alias_without_networks(self):
        x = np.zeros((3, 3, 54)); m = np.arange(72).reshape(3, 2, 3, 4) % 8
        s = np.arange(3); donor = np.full((3, 2, 4), 7, dtype=int)
        expected = np.concatenate((x, independent_route(m[:, 0], False), independent_route(m[:, 1], False)), -1)
        with mock.patch.object(c.core.base, 'actor_forward', side_effect=AssertionError('No network allowed')):
            outputs = [c.cross_time_whole(None, x, m, s, donor, False),
                       c.position_intervention(None, x, m, s, donor, [0, 1, 0], [1, 2, 3], False)]
        for got in outputs:
            np.testing.assert_array_equal(got['messages'], m)
            np.testing.assert_array_equal(got['action_inputs'], expected)
            self.assertTrue(got['reused_natural'])
            self.assertFalse(got['outward_patch_visible'])
            self.assertEqual(got['neural_forward_samples'], 0)
            self.assertEqual(got['neural_forward_calls'], 0)
            self.assertNotIn('action_indices', got)
            self.assertNotIn('action_probabilities', got)

    @mock.patch.object(c.core.base, 'actor_forward', side_effect=fake_forward)
    def test_wrong_receiver_history_is_not_silently_accepted(self, forward):
        x = np.zeros((1, 3, 54)); m = c.generate_messages(NETWORKS, x, True)['messages']
        donor = m[:, :, 0].copy()
        m[:, 1, 0, 0] = (m[:, 1, 0, 0]+1) % 8
        with self.assertRaisesRegex(ValueError, 'receiver-checkpoint natural history'):
            c.cross_time_whole(NETWORKS, x, m, 0, donor)

    def test_invalid_generation_inputs_rejected_before_forward(self):
        args = dict(networks=NETWORKS, observations=np.zeros((1, 3, 54)), live=True)
        bad = [('networks', None), ('networks', tuple(range(8))), ('live', 1),
               ('observations', np.zeros((0, 3, 54))), ('observations', np.zeros((1, 3, 53))),
               ('observations', np.full((1, 3, 54), np.nan)), ('include_routes', 1)]
        with mock.patch.object(c.core.base, 'actor_forward', side_effect=AssertionError('Invalid input reached network')):
            for key, value in bad:
                with self.subTest(key=key):
                    with self.assertRaises(ValueError):
                        c.generate_messages(**dict(args, **{key: value}))
