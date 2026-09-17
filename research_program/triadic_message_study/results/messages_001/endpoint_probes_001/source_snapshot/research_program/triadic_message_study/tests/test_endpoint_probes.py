"""Fake arithmetic backend only: no network construction, weight read or training."""
import unittest
from unittest.mock import patch
import numpy as np

from research_program.triadic_message_study import endpoint_probes as e
from research_program.triadic_message_study import runner


def fake_forward(which, x):
    # Synthetic discrete arithmetic chosen to expose both inter-window routes.
    module = which % 3
    if module == 0:
        categories = (x[:, 0].astype(int)[:, None] + np.arange(4)) % 8
        z = np.zeros((len(x), 4, 8))
        np.put_along_axis(z, categories[..., None], 4, axis=-1)
        return z.reshape(-1, 32), None
    routed = x[:, 54:150].reshape(-1, 3, 4, 8)
    received = (routed * np.arange(8)).sum(axis=(1, 2, 3)).astype(int)
    if module == 1:
        categories = (received[:, None] + np.arange(4)) % 8
        z = np.zeros((len(x), 4, 8))
        np.put_along_axis(z, categories[..., None], 4, axis=-1)
        return z.reshape(-1, 32), None
    second = x[:, 153:249].reshape(-1, 3, 4, 8)
    category = (received + (second*np.arange(8)).sum(axis=(1, 2, 3)).astype(int)) % 17
    z = np.zeros((len(x), 17))
    z[np.arange(len(x)), category] = 4
    return z, None


class EndpointProbeTests(unittest.TestCase):
    def setUp(self):
        self.networks = list(range(9))
        self.x = np.zeros((2, 3, 54))
        self.x[:, :, 0] = [[0, 1, 2], [0, 3, 2]]
        self.patcher = patch.object(runner.base, 'actor_forward', fake_forward)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_silent_deletions_exact_identity(self):
        natural = runner.rollout(self.networks, self.x, False)
        secondary = e.action_only_drop(self.networks, self.x, natural, runner)
        self.assertTrue(np.array_equal(natural['action_logits'], secondary['action_logits']))
        self.assertTrue(np.array_equal(natural['action_inputs'], secondary['action_inputs']))
        for constant in (False, True):
            generated = e.regenerate_from_first(self.networks, self.x, natural, False, runner, constant=constant)
            self.assertTrue(np.array_equal(generated['action_inputs'], natural['action_inputs']))
            self.assertTrue(np.array_equal(generated['action_logits'], natural['action_logits']))

    def test_full_drop_regenerates_second_window(self):
        natural = runner.rollout(self.networks, self.x, True)
        closed = runner.rollout(self.networks, self.x, False)
        secondary = e.action_only_drop(self.networks, self.x, natural, runner)
        self.assertTrue(np.array_equal(natural['messages'][:, 0], closed['messages'][:, 0]))
        self.assertFalse(np.array_equal(natural['messages'][:, 1], closed['messages'][:, 1]))
        self.assertTrue(np.array_equal(natural['messages'], secondary['messages']))
        self.assertFalse(np.array_equal(closed['action_inputs'], secondary['action_inputs']))
        generated = e.regenerate_from_first(self.networks, self.x, natural, True, runner)
        self.assertTrue(np.array_equal(generated['action_inputs'], closed['action_inputs']))

    def test_constant_body_keeps_visible_real_zero_symbols(self):
        natural = runner.rollout(self.networks, self.x, True)
        constant = e.regenerate_from_first(self.networks, self.x, natural, True, runner, constant=True)
        self.assertTrue(np.array_equal(constant['messages'][:, 0], natural['messages'][:, 0]))
        route = constant['action_inputs'][:, :, 54:153]
        body = route[:, :, :96].reshape(-1, 3, 3, 4, 8)
        self.assertTrue((route[:, :, 96:] == 1).all())
        for viewer in range(3):
            for sender in range(3):
                if viewer != sender:
                    self.assertTrue((body[:, viewer, sender, :, 0] == 1).all())
                    self.assertTrue((body[:, viewer, sender, :, 1:] == 0).all())
                else:
                    self.assertTrue(np.array_equal(body[:, viewer, sender].argmax(-1), natural['messages'][:, 0, sender]))

    def test_donor_transplant_both_directions(self):
        natural = runner.rollout(self.networks, self.x, True)
        for source, donor in ((0, 1), (1, 0)):
            got = e.transplant_listener(self.networks, self.x[source:source+1, 0],
                natural['messages'][source:source+1], natural['messages'][donor:donor+1], 0, True, runner)
            self.assertTrue(np.array_equal(got['action_inputs'], natural['action_inputs'][donor:donor+1, 0]))
            self.assertTrue(np.array_equal(got['action_probabilities'],
                e.distribution(natural['action_logits'][donor:donor+1, 0])))

    def test_own_w1_violation_rejected(self):
        natural = runner.rollout(self.networks, self.x, True)
        bad = natural['messages'][1:2].copy()
        bad[:, 0, 0, 0] = (bad[:, 0, 0, 0]+1) % 8
        with self.assertRaisesRegex(ValueError, 'Own first-window'):
            e.transplant_listener(self.networks, self.x[:1, 0], natural['messages'][:1], bad, 0, True, runner)

    def test_silent_pair_same_action(self):
        natural = runner.rollout(self.networks, self.x, False)
        self.assertTrue(np.array_equal(natural['action_logits'][0, 0], natural['action_logits'][1, 0]))

    def test_greedy_uses_probability_ties_exactly_like_runner(self):
        def near_tie(which, x):
            if which % 3 == 1:
                z = np.full((len(x), 4, 8), -1.0)
                z[:, :, 0], z[:, :, 1] = 0, 1e-20
                self.assertTrue((z.argmax(-1) == 1).all())
                self.assertTrue((e.distribution(z).argmax(-1) == 0).all())
                return z.reshape(-1, 32), None
            return fake_forward(which, x)
        with patch.object(runner.base, 'actor_forward', near_tie):
            natural = runner.rollout(self.networks, self.x, True)
            generated = e.regenerate_from_first(self.networks, self.x, natural, True, runner, constant=True)
            self.assertTrue((generated['messages'][:, 1] == 0).all())
            transplanted = e.transplant_listener(self.networks, self.x[:1, 0],
                natural['messages'][:1], natural['messages'][1:2], 0, True, runner)
            self.assertTrue((transplanted['own_second_messages'] == 0).all())

    def test_roundoff_scope_does_not_accept_action_change(self):
        e.compare_saved(np.array([1.0]), np.array([1.0+1e-13]), 'synthetic')
        with self.assertRaises(ValueError):
            e.compare_saved(np.array([1.0]), np.array([1.0+1e-8]), 'synthetic')
        with self.assertRaises(ValueError):
            e.compare_saved(np.array([float('nan')]), np.array([1.0]), 'synthetic')

    def test_independent_physical_score_all_4913_joint_actions_one_state(self):
        from itertools import product
        state = e.pairs.env.State((0, 6, 9), (2, 3, 0, 1))
        actions = np.array(list(product(range(17), repeat=3)), dtype=np.int16)
        reward, executed = e.settle_choices(np.tile(e.pairs.pack(state), (len(actions), 1)), actions)
        for k, row in enumerate(actions):
            actual = e.pairs.env.settle(state,
                {a: e.pairs.MENUS[i][row[i]] for i, a in enumerate(e.pairs.env.AGENTS)}, require_match=True)
            self.assertEqual(reward[k], actual['reward'])
            self.assertEqual(executed[k].tolist(),
                [actual['individual_feedback'][a]['executed'] for a in e.pairs.env.AGENTS])


if __name__ == '__main__':
    unittest.main()
